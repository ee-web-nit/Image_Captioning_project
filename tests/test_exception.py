from image_captioning.exception import CustomException
import sys

try:
    a = 10 / 0

except Exception as e:
    raise CustomException(e, sys)