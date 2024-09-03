#!/usr/bin/env python
# coding: utf-8

# In[1]:


number=123 
if number>99 and number<1000: 
    print('3 digit')
else: 
    print('Not 3 digit')



# In[3]:


response=input('Are you familiar with python ')
if response.upper()=="YES":
    print("You can skip this course :-)")
elif response.upper() == "NO":
    print("You are at the right place :-)")
else:
    print('Sorry wrong input :-(')


# In[7]:


for x in range(10): 
    print(x,end=' ')


# In[12]:


limit=int(input('Enter a limit:'))
sum=0
for i in range(1,limit+1):
    if i%2!=0:
        sum+=i
print("Odd sum="+str(sum))


# In[15]:


print(list(range(10))) 
print(list(range(1,10))) 
print(list(range(1,10,2)))


# In[18]:


number=int(input('Enter number:'))
s=0
while number>0:
    s+=number%10
    number=number//10
print(s)


# In[26]:


limit=int(input('Enter number:')) 
for num in range (2,limit+1): 
    is_divisible=False
    k=2
    while k<=num//2:
        if num%k==0:
            is_divisible=True 
            break;
        k+=1 
    if not is_divisible:
        print(num,end=' ')


# In[27]:


mylist=['a','b',1,1.2, True]
print(mylist) 
mylist.append("new")
print(mylist)
print(mylist.pop())
mylist.insert(2,'new')
print(mylist) 
mylist.remove('new')
print(mylist)

b=[1,2,3]
print(b)
mylist.append(b)
print(mylist)
mylist.remove(b)
print(mylist)
mylist.extend(b)
print(mylist) 
a=[2,3,1,4,5] 
a.sort()
print(a)

print(list('hello'))


# In[29]:


numbers=[0,1,2,3,4,5,6,7,8,9,10] 
print(numbers[1],numbers[-1])

sliced=numbers[5:11] 
print(sliced) 
slice1= numbers[5:] 
print(slice1) 
Sliced=numbers[:7] 
print(Sliced) 
slice2=numbers[-2:] 
print(slice2) 


# In[31]:


numbers=list(range(1,8))
print(numbers)
square=[]
for i in numbers: 
    square.append(pow(i,2)) 
    print(square)

square=[x**2 for x in numbers] 
print(square) 
odd_square=[x**2 for x in numbers if x%2!=0] 
print(odd_square)

A=[4,6,8,9] 
AxA=[(a,b) for a in A for b in A if a!=b] 
print(AxA)


# In[36]:


person={'name': 'Manu', 'age':28}
print(person['name'])
print('name' in person)
print('sex' in person)
person['sex']='male'
print(person)
for item in person:
    print(item,person[item])

for(key, value)in person.items(): 
    print(key.capitalize(),'\t:\t',value) 
print(person.keys())


# In[37]:


def square(number):
    return pow(number,2)
s=square(5)
print(s)


# In[41]:


def isPrime(number):
    for factor in range(2, (number//2)+1):
        if number%factor==0:
            return False
    return True
number=int(input('Enter the number'))
print(isPrime(number))


# In[49]:


def printPrimes(llimit, ulimit): 
    for num in range(llimit,ulimit+1): 
        if isPrime(num)==True: 
            print(num,end=' ') 

printPrimes(5,50)



# In[50]:


def swap(x,y):
    t=x
    x=y
    y=t
    return x,y
a=5
b=7
a,b=swap(a,b)
print(a,b)


# In[ ]:




