from typing import TypedDict
from pydantic import BaseModel , EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name : str = "abhishek"
    rollno : int
    age : Optional[str] = None
    mail : EmailStr
    cgpa : float = Field(gt = 0,lt = 10,default = 5,description = "A decimal value ") 
   

class Person(TypedDict):
    name : str 
    age : int 

new_student = {"rollno":78,"mail":"asmnxbc@gm.com"}

student = Student(**new_student)

new_person : Person = {"name":"abhishek","age":18}


print(student)

print(new_person)


