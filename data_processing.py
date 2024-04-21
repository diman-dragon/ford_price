import re
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder


def extract_vin_price_pairs(data):
    # Паттерн для поиска VIN-кода и цены
    pattern = re.compile(r'\[([A-Z0-9]{17})\:(\d+)\]')
    # Находим все соответствия паттерну в данных
    matches = pattern.findall(data)

    # Создаем список кортежей, содержащих VIN-коды и цены
    data_list = [(vin, int(price)) for vin, price in matches]

    # Создаем датафрейм из списка
    df = pd.DataFrame(data_list, columns=['VIN_code', 'price'])

    return df


# При проверке используем логику для проверки контрольной суммы VIN-кода по алгоритму Luhn.

# Определим функцию для проверки валидности VIN-кода

def is_valid_vin(vin):
    """
    Проверяет валидность VIN-кода.
    """
    if len(vin) != 17:
        return False
    if not re.match(r'^[A-HJ-NPR-Z0-9]+$', vin):
        return False
    
    # Проверка контрольной суммы
    transliteration = "0123456789.ABCDEFGH..JKLMN.P.R..STUVWXYZ"
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    checksum = 0
    for i in range(17):
        if transliteration.find(vin[i]) == -1:
            return False
        checksum += weights[i] * (transliteration.find(vin[i]) % 10)
    if vin[8].isdigit():
        expected = int(vin[8])
    elif vin[8] == 'X':
        expected = 10
    else:
        return False
    return (checksum % 11) == expected


def transform_data(data):
    # Выделение компонентов VIN-кода
    data['WMI_country'] = data['VIN_code'].str[0]  # Страна производства
    data['WMI_manufacturer'] = data['VIN_code'].str[1:3]  # Код производителя
    data['VDS_model'] = data['VIN_code'].str[3:6]  # Модель автомобиля
    data['VDS_body_type'] = data['VIN_code'].str[6]  # Тип кузова автомобиля
    data['VDS_engine_type'] = data['VIN_code'].str[7]  # Тип двигателя
    data['VDS_check_digit'] = data['VIN_code'].str[8]  # Контрольная цифра, используемая для проверки валидности VIN кода
    data['year_of_manufacture'] = data['VIN_code'].str[9]  # Год производства
    data['assembly_plant_code'] = data['VIN_code'].str[10]  # Код завода-изготовителя, на котором была собрана или завершена сборка автомобиля
    data['specific_vehicle_characteristics'] = data['VIN_code'].str[11]  # Специфические характеристики транспортного средства
    data['VIS_serial_number'] = data['VIN_code'].str[12:]  # Заводской номер автомобиля
    data.drop(['VIN_code', 'VDS_check_digit'], axis=1, inplace=True)
    
    data['VIS_serial_number'] = data['VIS_serial_number'].astype(int)

    # Создаем словарь для замены кодов стран на названия на английском языке
    country_mapping = {'2': 'Canada', '3': 'Mexico', '1': 'United States', '4': 'United States'}
    
    # Производим замену в столбце 'WMI_country'
    data['WMI_country'] = data['WMI_country'].replace(country_mapping)
    
    # Заменим известные года производства
    year_mapping = {
        'B': 2011, 'E': 2014, '7': 2007, 'D': 2013, 'C': 2012,
        '3': 2003, 'A': 2010, '5': 2005, '4': 2004, '9': 2009,
        '8': 2008, 'X': 1999, '6': 2006, 'Y': 2000, 'F': 2015,
        '2': 2002, 'W': 2001, 'V': 2002, '1': 2001, 'S': 1995,
        'T': 1996, 'R': 1994, 'H': 1987, 'K': 1989, 'P': 1993
    }
    
    # Заменяем значения в столбце 'year_of_manufacture' с использованием словаря
    data['year_of_manufacture'] = data['year_of_manufacture'].map(year_mapping)
    
    # Кодирование категориальных признаков по частотному принципу
    categorical_features = ['WMI_country', 'WMI_manufacturer', 
                            'VDS_model', 'VDS_body_type', 
                            'VDS_engine_type', 'assembly_plant_code', 
                            'specific_vehicle_characteristics']
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        data[feature] = label_encoders[feature].fit_transform(data[feature])
    
    # Масштабирование признаков
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled

