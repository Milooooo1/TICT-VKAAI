import os

percentages = []
for i in range(1, 100):
    out = os.popen(f"python3.10 main.py -k {i}").read()
    percentage = float(out[out.rfind('Accuracy: ')+10:out.rfind("%")])
    percentages.append(percentage)
    print(f"{percentage}% with k: {i}")

print(f"Max: {max(percentages)} with k: {percentages.index(max(percentages))}")