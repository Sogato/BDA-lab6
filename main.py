import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare_data(filename):
    """Загрузка и первичная подготовка данных из файла."""
    df = pd.read_csv(filename)
    print("Первые 3 строки исходных данных:")
    print(df.head(3))

    # Удаление столбца, не несущего полезной информации
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    print("\nДанные после удаления столбца 'Unnamed: 0':")
    print(df.head(3))

    return df


def analyze_nans(df):
    """Анализ пропущенных значений в данных."""
    # Проверка на наличие пропущенных значений
    NaNs_in_df = df.isnull()
    print("\nПроверка на NaNs, тип результата:")
    print(type(NaNs_in_df))
    print("\nПроверка на NaNs, первые 5 строк:")
    print(NaNs_in_df.head())

    # Подсчет количества пропущенных значений по столбцам
    NaNs_per_column = NaNs_in_df.sum()
    print("\nКоличество NaNs по столбцам:")
    print(NaNs_per_column.head())

    # Общее количество пропущенных значений
    NaNs_total = NaNs_per_column.sum()
    print(f"\nОбщее количество NaNs в данных: {NaNs_total}")

    # Процент пропущенных значений от общего числа
    NaNs_pct = np.round(NaNs_total / float(len(df) * len(df.columns)) * 100, decimals=4)
    print(f"\nDataFrame содержит: {NaNs_total} NaNs, что составляет {NaNs_pct}% от общего количества измерений")


def clean_data(df):
    """Очистка данных от пропущенных значений и преобразование типов."""
    df_clean = df.dropna().copy()  # Добавление .copy() для создания явной копии и избежания предупреждения
    print("\nПример преобразования строки в число:")
    str_val = '10.56'
    float_val = float(str_val)
    print(f"  Строковое значение: {str_val}, тип: {type(str_val)} \n  Числовое значение:  {float_val}, "
          f"тип после преобразования: {type(float_val)}")

    df_clean.loc[:, 'Ping (ms)_float'] = df_clean['Ping (ms)'].apply(lambda x: float(x))
    df_clean.loc[:, 'Download (Mbit/s)_float'] = df_clean['Download (Mbit/s)'].apply(lambda x: float(x))

    # Удаление столбцов без предупреждения
    df_clean = df_clean.drop(['Ping (ms)', 'Download (Mbit/s)'], axis=1)

    # Переименование столбцов без предупреждения
    df_clean = df_clean.rename(columns={'Ping (ms)_float': 'Ping (ms)', 'Download (Mbit/s)_float': 'Download (Mbit/s)'})

    # Переупорядочение столбцов для лучшей читаемости данных
    df_clean = df_clean.reindex(columns=['Date', 'Time', 'Ping (ms)', 'Download (Mbit/s)', 'Upload (Mbit/s)'])
    print("\nПервые 5 строк после очистки и переупорядочивания данных:")
    print(df_clean.head())

    return df_clean


def compute_statistics(df):
    """Вычисление и вывод статистических характеристик данных."""
    # Средние значения и стандартные отклонения
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)

    # Минимальные и максимальные значения
    mins = df.min()
    maxs = df.max()

    # Средние значения и стандартные отклонения
    print('\nСреднее время пинга: {:.2f} ± {:.2f} мс'.format(means['Ping (ms)'], stds['Ping (ms)']))
    print('Средняя скорость загрузки: {:.2f} ± {:.2f} Мбит/с'.format(means['Download (Mbit/s)'],
                                                                     stds['Download (Mbit/s)']))
    print('Средняя скорость отдачи: {:.2f} ± {:.2f} Мбит/с'.format(means['Upload (Mbit/s)'],
                                                                   stds['Upload (Mbit/s)']))

    # Минимальные и максимальные значения
    print('\nМинимальное время пинга: {:.2f} мс. Максимальное время пинга: {:.2f} мс'.format(mins['Ping (ms)'],
                                                                                           maxs['Ping (ms)']))
    print('Минимальная скорость загрузки: {:.2f} Мбит/с. Максимальная скорость загрузки: {:.2f} Мбит/с'.format(
        mins['Download (Mbit/s)'], maxs['Download (Mbit/s)']))
    print('Минимальная скорость отдачи: {:.2f} Мбит/с. Максимальная скорость отдачи: {:.2f} Мбит/с'.format(
        mins['Upload (Mbit/s)'], maxs['Upload (Mbit/s)']))


def plot_connection_stats(df):
    """Построение графика статистики интернет-соединения."""
    t = pd.to_datetime(df['Time'], format='%H:%M:%S')
    plt.figure(figsize=(10, 5))
    plt.plot(t, df['Ping (ms)'], 'o', label='Пинг (мс)')
    plt.plot(t, df['Download (Mbit/s)'], '-', label='Скорость загрузки (Мбит/с)')
    plt.plot(t, df['Upload (Mbit/s)'], '-', label='Скорость выгрузки (Мбит/с)')
    plt.legend(loc='upper right')
    plt.xlabel('Время', fontsize=16)
    plt.ylabel('Значение', fontsize=16)
    plt.title('Статистика интернет-соединения', fontsize=16)
    plt.savefig('connection_stats.png')


def plot_connection_stats_with_style(df):
    """Изменение стиля линий и добавление меток."""
    with plt.style.context('fivethirtyeight'):
        t = pd.to_datetime(df['Time'], format='%H:%M:%S')
        plt.figure(figsize=(10, 5))
        plt.plot(t, df['Ping (ms)'], 'o', label='Пинг (мс)')
        plt.plot(t, df['Download (Mbit/s)'], '-', label='Скорость загрузки (Мбит/с)')
        plt.plot(t, df['Upload (Mbit/s)'], '-', label='Скорость выгрузки (Мбит/с)')
        plt.legend(loc='upper right')
        plt.xlabel('Время', fontsize=16)
        plt.ylabel('Значение', fontsize=16)
        plt.title('Статистика интернет-соединения в стиле fivethirtyeight', fontsize=16)
        plt.savefig('connection_stats_styled.png')


def plot_histograms(df):
    """Представление данных в виде гистограмм."""
    plt.figure(figsize=(15, 5))

    # Гистограмма для пинга
    plt.subplot(1, 3, 1)  # 1 строка, 3 столбца, 1-й график
    df['Ping (ms)'].plot(kind='hist', bins=30, title='Пинг (мс)')
    plt.xlabel('Пинг (мс)')

    # Гистограмма для скорости загрузки
    plt.subplot(1, 3, 2)  # 1 строка, 3 столбца, 2-й график
    df['Download (Mbit/s)'].plot(kind='hist', bins=30, title='Скорость загрузки (Мбит/с)')
    plt.xlabel('Скорость загрузки (Мбит/с)')

    # Гистограмма для скорости выгрузки
    plt.subplot(1, 3, 3)  # 1 строка, 3 столбца, 3-й график
    df['Upload (Mbit/s)'].plot(kind='hist', bins=30, title='Скорость выгрузки (Мбит/с)')
    plt.xlabel('Скорость выгрузки (Мбит/с)')

    plt.tight_layout()  # Автоматическая корректировка подзаголовков для предотвращения наложения
    plt.savefig('connection_histograms.png')


def find_extreme_performance_times_and_values(df):
    """
    Нахождение времени и значений максимальной и минимальной скорости загрузки,
    выгрузки данных и задержки отправки эхо-запросов командой ping.
    """
    # Максимальная и минимальная скорость загрузки
    max_download_idx = df['Download (Mbit/s)'].idxmax()
    min_download_idx = df['Download (Mbit/s)'].idxmin()

    # Максимальная и минимальная скорость выгрузки
    max_upload_idx = df['Upload (Mbit/s)'].idxmax()
    min_upload_idx = df['Upload (Mbit/s)'].idxmin()

    # Максимальная и минимальная задержка отправки эхо-запросов командой ping
    max_ping_idx = df['Ping (ms)'].idxmax()
    min_ping_idx = df['Ping (ms)'].idxmin()

    print(f"\nМаксимальная скорость загрузки {df.loc[max_download_idx, 'Download (Mbit/s)']} Мбит/с была в "
          f"{df.loc[max_download_idx, 'Time']}, минимальная {df.loc[min_download_idx, 'Download (Mbit/s)']} Мбит/с - в "
          f"{df.loc[min_download_idx, 'Time']}.")
    print(f"Максимальная скорость выгрузки {df.loc[max_upload_idx, 'Upload (Mbit/s)']} Мбит/с была в "
          f"{df.loc[max_upload_idx, 'Time']}, минимальная {df.loc[min_upload_idx, 'Upload (Mbit/s)']} Мбит/с - в "
          f"{df.loc[min_upload_idx, 'Time']}.")
    print(f"Максимальная задержка пинга {df.loc[max_ping_idx, 'Ping (ms)']} мс была в "
          f"{df.loc[max_ping_idx, 'Time']}, минимальная {df.loc[min_ping_idx, 'Ping (ms)']} мс - в "
          f"{df.loc[min_ping_idx, 'Time']}.")


if __name__ == "__main__":
    # Загрузка и подготовка данных
    filename = 'rpi_data_compact.csv'
    df = load_and_prepare_data(filename)

    # Анализ пропущенных значений
    analyze_nans(df)

    # Очистка данных
    df_clean = clean_data(df)

    # Вычисление статистических характеристик
    compute_statistics(df_clean)

    # Выполнение функций построения графиков
    plot_connection_stats(df_clean)
    plot_connection_stats_with_style(df_clean)
    plot_histograms(df_clean)

    find_extreme_performance_times_and_values(df_clean)

    # Сохранение обработанных данных в новый файл
    df_clean.to_csv('rpi_data_processed.csv', index=False)

