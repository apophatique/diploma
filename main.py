import argparse
from controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='webcam', type=str)
parser.add_argument('-auditorium', default='8-207', type=str)
parser.add_argument('-date', default='2021.05.13', type=str)
parser.add_argument('-time', default='12:00', type=str)
args = parser.parse_args()


def main():
    controller = Controller()
    controller.run(
        args.mode,
        args.auditorium,
        args.date,
        args.time
    )


if __name__ == '__main__':
    main()
