import datetime

import requests


class ScheduleController:
    def __init__(self):
        self.request_url = 'https://rasp.omgtu.ru/api/'
        self.time_format = '%H:%M'

    def get_groups_in_auditorium(self, auditorium, date, time_str):
        auditorium_id = self.get_group_id(auditorium)
        search_part = 'schedule/auditorium/{auditorium_id}?start={date}&finish=2021.05.16&lng=1'.format(
            auditorium_id=auditorium_id,
            date=date
        )
        response = requests.get(self.request_url + search_part).json()

        '''
        # TODO
        print(response)
        return 'PIN-171'
        '''

        time_now = datetime.datetime.strptime(time_str, self.time_format)
        for lesson in response:
            begin_lesson_str = lesson['beginLesson']
            begin_lesson = datetime.datetime.strptime(begin_lesson_str, self.time_format)
            end_lesson_str = lesson['endLesson']
            end_lesson = datetime.datetime.strptime(end_lesson_str, self.time_format)

            if begin_lesson < time_now < end_lesson:
                if lesson['stream'] is None:
                    group = lesson['subGroup'].split('/')[0]
                    return group
                else:
                    stream_str = lesson['stream']
                    return stream_str

    def get_group_id(self, auditorium):
        search_part = 'search?term={0}&type=auditorium'.format(auditorium)
        response = requests.get(self.request_url + search_part).json()
        response = response[0]
        auditorium_id = response['id']
        return auditorium_id


if __name__ == '__main__':
    schedule_controller = ScheduleController()
    groups_in_auditorium = schedule_controller.get_groups_in_auditorium('8-207', '2021.05.10', '10:00')
    print(groups_in_auditorium)
