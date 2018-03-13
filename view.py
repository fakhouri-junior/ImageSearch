from flask import Flask, render_template, request, redirect,jsonify, url_for, flash
import urllib
import random, string
from flask_mail import Mail, Message
from model import Prediction, FinalResult
from core_classification import *
import time

import numpy as np
from multiprocessing.pool import ThreadPool


app = Flask(__name__)


"""
WORK WITH THE NUMPY ARRAY IN A THREAD, TRY TO MAKE IT READY FOR WORK BEFORE NEEDING IT
"""

pool = ThreadPool(processes=5)
async_result = pool.apply_async(load_array,())


@app.route('/')
@app.route('/index')
def homePage():
    return render_template("index.html")


@app.route('/api/v1/<path:url_path>')
def myAPI(url_path):

    # 2- download the image
    path_of_image = downloadImage(url_path)
    if path_of_image.startswith("Non"):
        return render_template('error.html')
    # 3- get the path of the image .... DONE IN STEP #2

    # 4- form the proper main method for core_classification
    result = mainWeb(path_of_image)

    # 5- display the output
    # 5'- slice the result into useful parts
    parts = result.split('\n')
    # 6- extract info from the parts to form the FinalResult object and jsonify it
    predictions = []
    for part in parts:
        part = part.strip()
        if part.startswith("Prediction") or part == "":
            continue
        myParts = part.split(' ')
        print myParts
        percentage = myParts[0]#percentage
        value = myParts[2]#value prediction
        prediction = Prediction(percentage,value)
        predictions.append(prediction)

    finalResult = FinalResult(url_path,predictions)
    return jsonify(Result=finalResult.serialize)



@app.route('/classify', methods=['POST'])
def classifyImage():
    if request.method == 'POST':
        # 1- get the url
        if request.form['url']:
            url_path = request.form['url']


            # 2- download the image
            path_of_image = downloadImage(url_path)
            if path_of_image.startswith("Non"):
                return render_template('error.html')
            # 3- get the path of the image .... DONE IN STEP #2

            # 4- form the proper main method for core_classification
            result = mainWeb(path_of_image)

            # 5- display the output
            # 5'- slice the result into useful parts
            # list of percentages
            predictions = []
            parts = result.split('\n')
            for part in parts:
                part = part.strip()
                if part.startswith("Prediction") or part == "":
                    continue
                myParts = part.split(' ')

                percentage = myParts[0]  # percentage
                value = myParts[2]  # value prediction
                prediction = Prediction(percentage, value)
                predictions.append(prediction)



            path_parts = path_of_image.split('/')
            betterPath = path_parts[5]+"/"+path_parts[6]+"/"+path_parts[7]

            """
            RETRIEVAL CODE GOES HERE
            """
            script_start_time = time.time()
            features_array = async_result.get()
            print 'async_result took %f ' % (time.time() - script_start_time,)

            list_of_paths = findSimilar([path_of_image], features_array)
            print list_of_paths

            # parse the location from disk to a location in server
            server_similar_images_parsed = parseImageSimilarPath(list_of_paths)

            # pass list_of_paths which represents the similar images to the template
            #return render_template('result.html', result=predictions, path_of_image=betterPath)

            return render_template('result.html', result=predictions, path_of_image=betterPath, similar=server_similar_images_parsed)

    else:
        return "REJECTED"


def downloadImage(url):
    # sanity checks
    # 1- check for http and .jpg or .png
    if url.startswith("http") and (url.endswith(".jpg") or url.endswith(".png")):

        resource = urllib.urlopen(url)
        jebrish = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in xrange(20))
        if url.endswith(".png"):
            output = open("/home/salim/PycharmProjects/ImageSearch/static/images/" + jebrish + ".png", "wb")
        else:
            output = open("/home/salim/PycharmProjects/ImageSearch/static/images/" + jebrish + ".jpg", "wb")

        output.write(resource.read())
        output.close()
        # return the path of the saved image
        return "/home/salim/PycharmProjects/ImageSearch/static/images/"+jebrish+".jpg"
    else:
        return "NonValid"

def parseImageSimilarPath(list_of_images):
    new_list = []
    for image_path in list_of_images:
        parts_slash = image_path.split('/')
        # '/home/salim/Downloads/test/tshirt1.jpg'
        name_of_image = parts_slash[5] # what we need
        new_list.append("/static/images_similar/"+name_of_image)

    return new_list


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.debug = True
    app.run(host = '0.0.0.0', port = 5000)