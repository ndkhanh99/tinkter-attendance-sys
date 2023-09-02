const schedule = require('node-schedule');
var admin = require("firebase-admin");
const { initializeApp, applicationDefault, cert } = require('firebase-admin/app');
const { getFirestore, Timestamp, FieldValue, Filter } = require('firebase-admin/firestore');
var serviceAccount = require("./key/attendace-sys-firebase-adminsdk-e2nde-ea40d5feeb.json");

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
});

const db = getFirestore();

const citiesRef = db.collection('subject');
let current = db.collection('current_subject');

const job = schedule.scheduleJob('*/1 * * * *', () => { // run every hour at minute 1
    citiesRef.get().then((result) => {
        if (result.empty) {
            console.log('No matching documents.');
            return;
        }

        result.forEach(doc => {
            // console.log(doc.id, '=>', doc.data());
            let subject = doc.data();
            console.log(subject.time_out);

            checkTime = new Date();
            console.log(checkTime - subject.time_in.toDate());

            let compare = checkTime - subject.time_in.toDate();
            if (compare > 0 && compare < 900000) {
                current.doc('current').update({
                    name: subject.name,
                    time_in: subject.time_in,
                    time_out: subject.time_out,
                }).then(console.log('updated current subect'));
            }
        });
    }).catch((err) => {
        console.log(err);
    });
});