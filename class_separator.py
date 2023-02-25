import os

entries = os.listdir('./archive')
size = len(entries)
i = 0
for entry in entries:
    if i % 1000 == 0:
        print(str(i) + " / " + str(size))
    file_class = entry.split('_')[0]
    if not os.path.exists("./archive/" + file_class):
        os.mkdir("./archive/" + file_class)
    os.rename("./archive/" + entry, "./archive/" + file_class + "/" + entry)
    i += 1
