import csv

# 중첩 리스트
data_list = [['Name', 'Age', 'City'],
        ['John Doe', '30', 'New York'],
        ['Jane Doe', '25', 'Chicago']]

# 딕셔너리 리스트
dict_list = [{"name": "John", "age": 30, "city": "New York"},
             {"name": "Mike", "age": 28, "city": "Los Angeles"},
             {"name": "Sara", "age": 25, "city": "Chicago"}]

def write_csv_dict(dict_list, file_name = "dict.csv"):
    # CSV 파일로 저장
    headers = ["text"]
    with open(file_name, 'w', newline = "", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for dict_obj in dict_list:
            writer.writerow(dict_obj)

def write_csv_list(new_list ,file_name = "list.csv"):
    with open(file_name, 'w', newline = "", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(new_list)

if __name__ == "__main__":
    write_csv_dict(dict_list)
    write_csv_list(data_list)

