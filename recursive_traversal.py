# Python program to explain shutil.move() method  
      
# importing os module  
import os 
  
# importing shutil module  
import shutil  
  
# path  
path = './'
print(path)  
# List files and directories  
# in 'C:/Users/Rajnish/Desktop/GeeksforGeeks'  
print("Before moving file:")  
print(os.listdir(path))  

for k in os.listdir(path):
    if "lesson" in k:
        print(k)
        k = k + "/"
        source = os.path.join(path,k)
        #os.system("mkdir "+path+k+"v3/" )
        destination = os.path.join(path,"v3/",k)
        # print(source)
        # print(destination)
        try:
            dest = shutil.move(source, destination, copy_function = shutil.copytree)
        except Exception as e:
            print(e)



  
"""  
# Source path  
source = 'C:/Users/Rajnish/Desktop/GeeksforGeeks/source'
  
# Destination path  
destination = 'C:/Users/Rajnish/Desktop/GeeksforGeeks/destination'
  
# Move the content of  
# source to destination  
dest = shutil.move(source, destination)  
  
# List files and directories  
# in "C:/Users / Rajnish / Desktop / GeeksforGeeks"  
print("After moving file:")  
print(os.listdir(path))  
  
# Print path of newly  
# created file  
print("Destination path:", dest)  
"""