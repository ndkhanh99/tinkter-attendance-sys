import firebase_admin
from firebase_admin import credentials, firestore
import datetime
from datetime import timedelta
import pytz

utc = pytz.UTC
cred = credentials.Certificate(
    "key/attendace-sys-firebase-adminsdk-e2nde-ea40d5feeb.json")
firebase_admin.initialize_app(cred)


names = set()
db = firestore.client()

# db.collection('test').document('duykhanh').set({'name': 'duykhanh', 'age': 25})

# result = db.collection('subject').document('subject').get()
# if result.exists:
# print(result.to_dict())
# result = result.to_dict()
# print(result['date_in'])
# today = datetime.datetime.now()
# print(today)

# print(utc.localize(today) < result['date_in'])

# for i in result:
#     print(i, result[i])
# print(result.date_in)

result = db.collection(
    'current_subject').document('current').get()
result = result.to_dict()
print(result['name'])

# time_compare = result['time_in'] + timedelta(hours=7)
# print(time_compare)
# today = datetime.datetime.now()
# print(today)
# status = 'in_time'
# if utc.localize(today) > time_compare:
#     status = 'late'
#     print('vo tre')
# else:
#     print('dung gio')


# result = db.collection('users').document(
#     '1911368').get()
# if result.exists:
#     print(result.to_dict())
# else:
#     print('no match')

# checkExist = db.collection('check_in').where(
#     "finger_id", "==", '2').get()
# print(len(checkExist))
