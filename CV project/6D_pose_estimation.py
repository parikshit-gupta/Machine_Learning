import torch
import torch.nn as nn
import torch.nn.functional as F
import timm     # pytorch image models
from tqdm import tqdm      # library to show progress bars


"""
    Differences from the implementation in the given paper:
    1) No data augumentation pipeline (LM and LMO files were not opening)
    2) The encoder and decoder used are off the shelf from the pytorch library (decoder archietecture was not explicitly mentioned in the paper)
    3) The authors used a two phased training: a) proxy reconstruction (autoencoder type training)
                                               b) Multi task learning (fine-tuning initialised weights from autoencoder)
            BUT due to lack of data to work on, we coded only one training_epoch for MTL.
"""


"""
    MODEL DEFINATION
"""

class ViTEncoder(nn.Module):
    """
    Wrapper around timm ViT to expose patch tokens and class token.
    Uses timm.create_model(..., num_classes=0) and .forward_features
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=False):
        super().__init__()      # super enables subclass to inherit parent methods
        
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        #by default the vision encoder outputs classes, but here we want the RAW encodings outputted by
        #the encoder- to be used for the proxy reconstruction task and 6D pose regression
        
        # embed_dim exposed by timm models
        self.embed_dim = self.vit.embed_dim     #internal embedding dimension of VIT encoder is 768
        
        self.patch_size = getattr(self.vit, 'patch_size', 16)
        self.img_size = 224
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

    def forward(self, x):  # x: (B,3,224,224)
        # timm ViT typical forward_features returns (B, N+1, D)
        tokens = self.vit.forward_features(x)   # tokenisation, positional encoding
        cls_token = tokens[:, 0]  # (B, D)      # class token
        patches = tokens[:, 1:, :]  # (B, N, D)
        return cls_token, patches  # note variable names: cls token -> cls_token

class TransformerDecoderRecon(nn.Module):
    """
    Transformer decoder that maps encoded patch tokens -> per-patch embeddings suitable for reconstruction.
    Implemented with nn.TransformerDecoder for simplicity.
    """
    
    """enc_dim: dimensionality of encoder outputs (768 here)
        dec_dim: dimensionality inside decoder (512 by default).
        dec_depth: number of transformer layers (8 → quite deep).
        nhead: number of attention heads per layer (8).
    """
    def __init__(self, enc_dim=768, dec_dim=512, dec_depth=8, nhead=8, patch_size=16, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid
        
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)   #changes token dimensions for every token from 768 to 512
        
        #defining one layer of the decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=dec_dim, nhead=nhead, dim_feedforward=dec_dim*4, batch_first=True)
        
        #stacking dec_depth such layers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_depth)
        
        # query embeddings (learnable) for the decoder (one query per patch)
        """A query in a decoder is a learnable embedding that asks for information from the encoder to reconstruct 
        or generate the missing parts of the output —in IRPE, each query corresponds to a patch of the image the 
        model needs to reconstruct.
        """
        self.query_embed = nn.Parameter(torch.zeros(1, self.n_patches, dec_dim))
        nn.init.normal_(self.query_embed, std=0.02)

        # heads that map decoder output per patch to pixels (patch_size*patch_size*3)
        self.patch_to_rgb = nn.Linear(dec_dim, patch_size*patch_size*3)
        self.patch_to_mask = nn.Linear(dec_dim, patch_size*patch_size*1)

        """ Think of each decoder layer as a “conversation”:
            First, each patch query talks to other patch queries (self-attn),
            Then, it asks the encoder for more information (cross-attn),
            Finally, it reflects and transforms that information internally (feed-forward).
        """
    def forward(self, enc_patches):
        # enc_patches: (B, N, D_enc)
        dec_in = self.enc_to_dec(enc_patches)  # (B, N, D_dec)
        B = dec_in.shape[0]
        # expand queries
        q = self.query_embed.expand(B, -1, -1).contiguous()  # (B, N, D_dec)
        # transformer decoder: queries attend to encoder outputs
        dec_out = self.decoder(tgt=q, memory=dec_in)  # (B, N, D_dec)
        rgb_patches = self.patch_to_rgb(dec_out)  # (B, N, patch_pixels*3)
        mask_patches = self.patch_to_mask(dec_out)  # (B, N, patch_pixels*1)
        # reshape handled externally
        return rgb_patches, mask_patches

def reassemble_patches(patches, patch_size=16, img_size=224, channels=3):
    """
    patches: (B, N, patch_size*patch_size*C)
    returns image: (B, C, img_size, img_size)
    """
    B, N, P = patches.shape
    grid = img_size // patch_size
    patches = patches.view(B, N, channels, patch_size, patch_size)
    out = patches.new_zeros((B, channels, img_size, img_size))
    idx = 0
    for i in range(grid):
        for j in range(grid):
            out[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[:, idx]
            idx += 1
    return out

class PoseHead(nn.Module):
    """
    Pose head that maps class token to rot6d (6) and translation (ox, oy, tnz)
    """
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),      #the paper used GELU
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True)
        )
        self.rot = nn.Linear(hidden//2, 6)
        self.trans = nn.Linear(hidden//2, 3)

    def forward(self, cls_token):
        h = self.mlp(cls_token)
        r6 = self.rot(h)
        t = self.trans(h)
        return r6, t

class IRPEFull(nn.Module):
    """
    Full model combining encoder, transformer-decoder reconstructor, and pose head.
    Accepts 4-channel input: rgb_crop + reconstruction marker.
    """
    def __init__(self, vit_name='vit_base_patch16_224', pretrained=False, decoder_dim=512, decoder_depth=8):
        super().__init__()
        # small input proj 4->3
        self.input_proj = nn.Conv2d(4, 3, kernel_size=1)
        # encoder
        self.encoder = ViTEncoder(model_name=vit_name, pretrained=pretrained)
        # decoder
        self.decoder = TransformerDecoderRecon(enc_dim=self.encoder.embed_dim, dec_dim=decoder_dim, dec_depth=decoder_depth,
                                               patch_size=self.encoder.patch_size, img_size=self.encoder.img_size)
        # pose head
        self.pose_head = PoseHead(in_dim=self.encoder.embed_dim)
        # store metadata
        self.patch_size = self.encoder.patch_size
        self.img_size = self.encoder.img_size
        self.grid = self.img_size // self.patch_size

    def forward(self, x4c):
        # x4c: (B,4,H,W)
        x = self.input_proj(x4c)  # (B,3,H,W)
        cls_token, patch_tokens = self.encoder(x)  # cls_token: (B,D), patch_tokens: (B,N,D)
        # reconstruction
        rgb_patches, mask_patches = self.decoder(patch_tokens)  # (B,N,P*3), (B,N,P*1)
        recon_rgb = reassemble_patches(rgb_patches, patch_size=self.patch_size, img_size=self.img_size, channels=3)
        recon_mask = reassemble_patches(mask_patches, patch_size=self.patch_size, img_size=self.img_size, channels=1)
        recon_mask = torch.sigmoid(recon_mask)
        # pose
        rot6d, trans = self.pose_head(cls_token)
        return recon_rgb, recon_mask, rot6d, trans


"""
    UTILITY FUNCTIONS
"""

def make_reconstruction_marker(vis_mask: torch.Tensor, patch_size=16, low=0.0, high=0.5):
    """
    vis_mask: (B,1,H,W) float/binary
    returns marker: (B,1,H,W) binary float
    For each patch, compute visible ratio r; sample tau~U(low,high); marker=1 if r > tau else 0
    """
    B, C, H, W = vis_mask.shape
    assert C == 1
    ph = patch_size
    assert H % ph == 0 and W % ph == 0
    nh = H // ph
    nw = W // ph
    device = vis_mask.device
    marker = torch.zeros_like(vis_mask)
    for i in range(nh):
        for j in range(nw):
            patch = vis_mask[:, :, i*ph:(i+1)*ph, j*ph:(j+1)*ph]
            ratio = patch.view(B, -1).mean(dim=1)  # [B]
            rand_val = torch.rand(B, device=device) * (high - low) + low
            keep = (ratio > rand_val).float().view(B, 1, 1, 1)
            marker[:, :, i*ph:(i+1)*ph, j*ph:(j+1)*ph] = keep
    return marker

def rot6d_to_rotmat(x):
    """
    x: (B,6)
    returns R: (B,3,3)
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    # make a2 orthogonal to b1
    proj = (b1 * a2).sum(dim=1, keepdim=True)
    b2_ = a2 - proj * b1
    b2 = F.normalize(b2_, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    R = torch.stack([b1, b2, b3], dim=2)  # columns are b1,b2,b3
    return R

# small helper to compute Frobenius / L1 between rot matrices
def rotation_loss_matrix(R_pred, R_gt, loss_type='l1'):
    if loss_type == 'l1':
        return F.l1_loss(R_pred, R_gt)
    else:
        return F.mse_loss(R_pred, R_gt)
    

"""
   LOSS FUNCTIONS
"""

class IRPELoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_mask=1.0, lambda_R=0.1, lambda_tx=10.0, lambda_tz=100.0):
        
        # constants are defined emperically by the authors of the paper
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_mask = lambda_mask
        self.lambda_R = lambda_R
        self.lambda_tx = lambda_tx
        self.lambda_tz = lambda_tz
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    def forward(self, outputs, targets):
        """
        outputs: (recon_rgb, recon_mask, rot6d_pred, trans_pred)
        targets dict has keys 'rgb_render', 'amodal_mask', 'R_gt', 't_gt'
        """
        recon_rgb, recon_mask, rot6d_pred, trans_pred = outputs
        rgb_gt = targets['rgb_render']
        mask_gt = targets['amodal_mask']
        R_gt = targets['R_gt']
        t_gt = targets['t_gt']

        loss_rec = self.mse(recon_rgb, rgb_gt)
        loss_mask = self.bce(recon_mask, mask_gt)

        R_pred = rot6d_to_rotmat(rot6d_pred)  # (B,3,3)
        loss_R = rotation_loss_matrix(R_pred, R_gt, loss_type='l1')

        loss_tx = self.l1(trans_pred[:, 0:2], t_gt[:, 0:2])
        loss_tz = self.l1(trans_pred[:, 2], t_gt[:, 2])

        loss = (self.lambda_rec * loss_rec + self.lambda_mask * loss_mask +
                self.lambda_R * loss_R + self.lambda_tx * loss_tx + self.lambda_tz * loss_tz)

        return {'total': loss, 'rec': loss_rec, 'mask': loss_mask, 'R': loss_R, 'tx': loss_tx, 'tz': loss_tz}
    
    
"""
    TRAINING
    
    NOTE:  we have not coded the loader method, it is a dataLoader that yields batches of training samples.
"""

#this function will be called from inside the main training loop (not explicitly coded here)
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, verbose=True):
    model.train()
    pbar = tqdm(loader, desc=f"Train {epoch}")
    losses_accum = {}
    for batch in pbar:
        rgb_crop = batch['rgb_crop'].to(device)
        vis_mask = batch['vis_mask'].to(device)
        rgb_render = batch['rgb_render'].to(device)
        amodal_mask = batch['amodal_mask'].to(device)
        R_gt = batch['R_gt'].to(device)
        t_gt = batch['t_gt'].to(device)

        # marker generation
        marker = make_reconstruction_marker(vis_mask, patch_size=model.patch_size).to(device)
        x4c = torch.cat([rgb_crop, marker], dim=1)

        recon_rgb, recon_mask, rot6d, trans = model(x4c)

        targets = {'rgb_render': rgb_render, 'amodal_mask': amodal_mask, 'R_gt': R_gt, 't_gt': t_gt}
        losses = criterion((recon_rgb, recon_mask, rot6d, trans), targets)
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # logging
        if verbose:
            pbar.set_postfix({'loss': losses['total'].item(), 'rec': losses['rec'].item()})
    return

