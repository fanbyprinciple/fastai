# this will create a copy of readme file 
# with modifications of your choice

f = open('readme.md','r')

all_lines = f.readlines()

new_lines = []
for i in all_lines:
    if i[0] == "#":
        print(i)
        new_lines.append(i+"-------")
    else:
        new_lines.append(i)

o = open('readme_copy.md', 'w+')

o.write("".join(new_lines))