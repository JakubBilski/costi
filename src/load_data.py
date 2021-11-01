from pathlib import Path


def read_x(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    examples = []
    for l in lines:
        info_line = "".join(l.split("\n")).split(" ")
        example_id = int(info_line[0])
        while example_id >= len(examples):
            examples.append([])
        channel = int(info_line[1]) - 1
        start_timestamp = float(info_line[2])
        end_timestamp = float(info_line[3])
        if len(info_line) == 4:
            value = 1
        else:
            value = float(info_line[4])
        examples[example_id].append([start_timestamp, end_timestamp, channel, value])

    return examples

def read_y(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        y = []
        for i in range(len(lines)):
            y.append(int(lines[i]))
    return y

def load_auslan2():
    x = read_x(Path('data/constrained/AUSLAN2/data.txt'))
    y = read_y(Path('data/constrained/AUSLAN2/classes.txt'))
    return x, y

def load_blocks():
    x = read_x(Path('data/constrained/BLOCKS/data.txt'))
    y = read_y(Path('data/constrained/BLOCKS/classes.txt'))
    return x, y

def load_context():
    x = read_x(Path('data/constrained/CONTEXT/data.txt'))
    y = read_y(Path('data/constrained/CONTEXT/classes.txt'))
    return x, y

def load_hepatitis():
    x = read_x(Path('data/constrained/HEPATITIS/data.txt'))
    y = read_y(Path('data/constrained/HEPATITIS/classes.txt'))
    return x, y

def load_pioneer():
    x = read_x(Path('data/constrained/PIONEER/data.txt'))
    y = read_y(Path('data/constrained/PIONEER/classes.txt'))
    return x, y

def load_skating():
    x = read_x(Path('data/constrained/SKATING/data.txt'))
    y = read_y(Path('data/constrained/SKATING/classes.txt'))
    return x, y

def load_musekey_constrained():
    x = read_x(Path('data/constrained/MUSEKEY_CON/data.txt'))
    y = read_y(Path('data/constrained/MUSEKEY_CON/classes.txt'))
    return x, y

def load_weather_constrained():
    x = read_x(Path('data/constrained/WEATHER_CON/data.txt'))
    y = read_y(Path('data/constrained/WEATHER_CON/classes.txt'))
    return x, y

def load_musekey_unconstrained():
    x = read_x(Path('data/unconstrained/MUSEKEY/data.txt'))
    y = read_y(Path('data/unconstrained/MUSEKEY/classes.txt'))
    return x, y

def load_weather_unconstrained():
    x = read_x(Path('data/unconstrained/WEATHER/data.txt'))
    y = read_y(Path('data/unconstrained/WEATHER/classes.txt'))
    return x, y
