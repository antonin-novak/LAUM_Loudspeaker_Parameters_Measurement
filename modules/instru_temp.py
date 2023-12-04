#!/usr/bin/python3
# lebon 07/11/2017

import requests


def get_temperature_humidity():
    try:
        # variables
        # give the ip address of the room
        domoticzserver = "laum-raspv3-7.univ-lemans.fr:8080"
        device_id = '15' 						# give the id of the device
        user = 'laum'
        password = 'laum'

        # method to send a request to the database

        def domoticzrequest(url):
            response = requests.get(
                url, auth=requests.auth.HTTPBasicAuth(user, password))
            return response.json()

        # MAIN
        # make the database request
        domoticzurl = "http://" + domoticzserver + \
            "/json.htm?type=devices&rid=" + device_id
        # get data from database
        json_object = domoticzrequest(domoticzurl)
        if json_object["status"] == "OK":
            if json_object["result"][0]["idx"] == device_id:
                device_data = json_object["result"][0]["Data"]
        # print temperature and humidity

        temperature_humidity = device_data

    except:
        print('temperature/humidity sensor not accessible')
        temperature_humidity = None

    return temperature_humidity


if __name__ == '__main__':
    print(get_temperature_humidity())
