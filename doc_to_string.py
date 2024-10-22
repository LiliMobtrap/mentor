file_name = "bap.txt"

with open(file_name, 'r', encoding='utf-8') as file:
    string = file.read().replace('\n', ' ')
