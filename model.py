class Prediction(object):

    def __init__(self, percentage, value):
        self.percentage = percentage
        self.value = value

    def getPercentage(self):
        return self.percentage

    def getValue(self):
        return self.value

class FinalResult(object):

    def __init__(self, url, pred):
        self.url = url
        self.pred = pred

    @property
    def serialize(self):
        return {
            'url': self.url,
            '1': {'percentage': self.pred[0].getPercentage(),
                  'prediction': self.pred[0].getValue()},
            '2': {'percentage': self.pred[1].getPercentage(),
                  'prediction': self.pred[1].getValue()},
            '3': {'percentage': self.pred[2].getPercentage(),
                  'prediction': self.pred[2].getValue()},
            '4': {'percentage': self.pred[3].getPercentage(),
                  'prediction': self.pred[3].getValue()},
            '5': {'percentage': self.pred[4].getPercentage(),
                  'prediction': self.pred[4].getValue()},
        }