
import ssl
import urllib.request

import json

import datetime


class CWMSDownloader:
    #def __init__(self, url="http://cwms-data.usace.army.mil/cwms-data/timeseries?"):
    #def __init__(self, url="https://water.usace.army.mil/cwms-data/timeseries?"):
    def __init__(self, url="http://water.usace.army.mil/cwms-data/timeseries?"):
        """Constructor for CWMS Downloader class"""
        self.service_url = url

    def get_data(self, office, name, begin, data_format):
        """ Download data for request office and station name with
        data range determined by begin and file format controled by data_format. """

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # build the request string out of parameters
        request_str = self.service_url\
            + "office="+office \
            + "&name="+name\
            + "&begin="+begin\
            + "&format="+data_format

        request_str = urllib.parse.quote(request_str, ":/?&=")

        print( datetime.datetime.now(), end = " --- " )
        print(request_str)

        # get the response from the web service
        resp = urllib.request.urlopen(request_str, context=ctx)

        # print(resp)

        if data_format == "json":
            # extract the response data as a byte array
            resp_data = resp.read()

            # print(type(resp_data))
            # print(resp_data)

            # convert the byte array to string
            json_resp = resp_data.decode('utf8')

            # print(type(json_resp))
            # print(json_resp)

            json_data = json.loads(json_resp)

            # s = json.dumps(json_data, indent=4, sort_keys=True)
            # print(s)

            return json_data
        else:
            # extract the response data as a byte array
            resp_data = resp.read()

            # print(type(resp_data))
            # print(resp_data)

            # convert the byte array to string
            txt_resp = resp_data.decode('utf8')

            return txt_resp


    def get_data_range(self, office, name, begin, end, data_format):
        """ Download data for request office and station name with
        data range determined by begin and file format controled by data_format. """

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # build the request string out of parameters
        request_str = self.service_url\
            + "office="+office \
            + "&name="+name\
            + "&begin="+begin\
            + "&end="+end\
            + "&format="+data_format

        request_str = urllib.parse.quote(request_str, ":/?&=")

        print( datetime.datetime.now(), end = " --- " )
        print(request_str)

        # get the response from the web service
        resp = urllib.request.urlopen(request_str, context=ctx)

        # print(resp)

        if data_format == "json":
            # extract the response data as a byte array
            resp_data = resp.read()

            # print(type(resp_data))
            # print(resp_data)

            # convert the byte array to string
            json_resp = resp_data.decode('utf8')

            # print(type(json_resp))
            # print(json_resp)

            json_data = json.loads(json_resp)

            # s = json.dumps(json_data, indent=4, sort_keys=True)
            # print(s)

            return json_data
        else:
            # extract the response data as a byte array
            resp_data = resp.read()

            # print(type(resp_data))
            # print(resp_data)

            # convert the byte array to string
            txt_resp = resp_data.decode('utf8')

            return txt_resp
