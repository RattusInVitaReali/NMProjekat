import os

i = 0
classes = os.listdir('./archive')
for brand in classes:
    for file in os.listdir('./archive/' + brand):
        if i % 5 == 0:
            if not os.path.exists("./testing/" + brand):
                os.mkdir("./testing/" + brand)
            os.rename("./archive/" + brand + "/" + file, "./testing/" + brand + "/" + file)
        i += 1
