import os

import matplotlib.pyplot as plt

brands = []

counts = []


for brand in os.listdir("./archive"):
    brands.append(brand)
    counts.append(len(os.listdir("./archive/" + brand)))

plt.bar(brands, counts)

plt.xlabel("Brands")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
