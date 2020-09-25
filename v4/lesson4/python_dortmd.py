f = open('./readme.md', 'r+')
new_readme = []
new_lines = []
for i in f.readlines():
    new_lines = []
    for j in range(len(i)):
        new_lines.append(i[j])
        if(i[0]=="!"):
            if(i[j]=='('):
                new_lines.append("./img/")
                
    new_readme.append("".join(new_lines))

print("".join(new_readme))
