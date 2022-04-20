import os

percentages = []
for i in range(1, 366):
    out = os.popen(f"python3.10 main.py {i}").read()
    percentage = float(out[out.find(': ')+2:out.rfind("%")])
    percentages.append(percentage)
    print(f"{percentage}% with k: {i}")

print(f"Max: {max(percentages)} with k: {percentages.index(max(percentages))}")