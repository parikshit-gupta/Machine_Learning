class student: 
    name=int(10)
    print("hello")
    def test(self):
        self.var1=111
class int:
    print("hello1")
#name=int(input("enter ur name: "))
name=10
print(name)
name="name"
print(name)

a=student()
#a.name="ten"
a.sibling1=9
a.sibling1_name="nine"
a.sibling2_name="eleven"
a.sibling2=11
print(a.__dict__)
print(student.__dict__)

s1=student()
o1=student()

c=100
print(a.__dict__)
def func_c(a):
    a.sibling1=8
    global pot
    pot=98
    print(c,"potty")
    print(pot)
    print(a.__dict__)
    return

def func_c(b):
    b.sibling1=7
    pg="hemlom"
    global pot
    pot=56
    print(b.__dict__)
    return

func_c(a)
print(pot)
print(a.__dict__)

print(name.__class__)
print(type(name))

h=10
#print(h.__dict__)


class student1:
    totalStudents=20
    __pottyStudents=70#private class attribute
    _hemlomStudents=90#protected class attribute
    
    def F1(self):
        print(student1.totalStudents)
        
    def F2(self):
        print(self.totalStudents)
        
    def F3(self):
        print(self.__pottyStudents)
        
    @classmethod
    def F4(cls):
        print(student1.totalStudents)
    
    @staticmethod
    def F5():
        student1.totalStudents=1000
        print(student1.totalStudents)

x1=student1()
x1.F5()
#we can create a new instance attribute which is not private, with the same name as a private attribute
#now outside the class refering to that name will refer to the instance attribute created
x1.__pottyStudents=80
print(x1.totalStudents)
print(x1.__pottyStudents)
x1.F3()
print(student1.__dict__)
#print(x1.__pottyStudents)
x1.totalStudents=30
print(x1.totalStudents)

x1.F1()
x1.F2()
student1.F2(student1())
student1().F2()

k=90

def foo():
    print("hemlom")
    k=10
    print(k)

    def potty():
        nonlocal k
        k=20
        print("potty")
        pass
    potty()
    print(k)

foo.__call__()
print(k)


class employee:
    def __init__(self, name, age):
        self.name=name
        self.age=age
    
    @classmethod
    #the current class's name is passed to cls
    def using_DOB(cls, name, YOB):
        #this returns an instance of the class in which the class method is defined
        return cls(name, 2024-YOB)#this line calls the constructor of the class employee to create an instance
    
    @staticmethod
    def add(a,b):
        print(a+b)
        
    def show(self):
        print("name: ", self.name, "age: ",self.age)
        employee.add(4,5)
        

   
e1=employee("pg", 19)
#using a class method to create an object of class employee
e2=employee.using_DOB("vg",2008)
e1.show()
e2.show()

#add(4,5) static method not accessible outside the class

