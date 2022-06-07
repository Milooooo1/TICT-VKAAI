import csv

def main():
    with open('RondeTafel.csv', 'r', newline='') as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
        for row in csv_data:
            print(', '.join(row))

if __name__ == "__main__":
    main()