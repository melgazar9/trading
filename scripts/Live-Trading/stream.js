var fs = require('fs');

// Utility
    function jsonToQueryString(json) {
        return Object.keys(json).map(function(key) {
                return encodeURIComponent(key) + '=' +
                    encodeURIComponent(json[key]);
            }).join('&');
    }

// Fill in credentials below
var userPrincipalsResponse = JSON.parse(fs.readFileSync('/home/melgazar9/Trading/TD/Live-Trading/credentials_config/credentials.json', 'utf8'));


    //Converts ISO-8601 response in snapshot to ms since epoch accepted by Streamer
    var tokenTimeStampAsDateObj = new Date(userPrincipalsResponse.streamerInfo.tokenTimestamp);
    var tokenTimeStampAsMs = tokenTimeStampAsDateObj.getTime();


var credentials = {
    "userid": userPrincipalsResponse.accounts[0].accountId,
    "token": userPrincipalsResponse.streamerInfo.token,
    "company": userPrincipalsResponse.accounts[0].company,
    "segment": userPrincipalsResponse.accounts[0].segment,
    "cddomain": userPrincipalsResponse.accounts[0].accountCdDomainId,
    "usergroup": userPrincipalsResponse.streamerInfo.userGroup,
    "accesslevel": userPrincipalsResponse.streamerInfo.accessLevel,
    "authorized": "Y",
    "timestamp": tokenTimeStampAsMs,
    "appid": userPrincipalsResponse.streamerInfo.appId,
    "acl": userPrincipalsResponse.streamerInfo.acl
}

var request = {
    "requests": [
            {
                "service": "ADMIN",
                "command": "LOGIN",
                "requestid": 0,
                "account": userPrincipalsResponse.accounts[0].accountId,
                "source": userPrincipalsResponse.streamerInfo.appId,
                "parameters": {
                    "credential": jsonToQueryString(credentials),
                    "token": userPrincipalsResponse.streamerInfo.token,
                    "version": "1.0"
                }
            }
    ]
}

// The below code allows connection to the server via shell - this way I can log the data
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8978 });
wss.on('connection', function connection(ws, req) {
  const ip = req.connection.remoteAddress;
});


var mySock = new WebSocket("wss://" + userPrincipalsResponse.streamerInfo.streamerSocketUrl + "/ws");


mySock.onmessage = function(evt) { console.log(evt.data); }; mySock.onclose = function() { console.log("CLOSED"); };

setTimeout(function(){
	mySock.send(JSON.stringify(request));
	setTimeout(function(){
		var request = {
			"requests": [
					{

            // Live Stream

            // "service": "LEVELONE_FUTURES",
            // "requestid": "2",
            // "command": "SUBS",
            // "account": userPrincipalsResponse.accounts[0].accountId,
            // "source": userPrincipalsResponse.streamerInfo.appId,
            // "parameters": {
            //     "keys": "/CLV8",
            //     "fields": "0,1,2,3,4,5,8,11,13,14,18,19,20,23,27,29,30,32,34,35"

            // Live Stream Data - 1min
            // "service": "CHART_FUTURES",
            // "requestid": "2",
            // "command": "SUBS",
            // "account": userPrincipalsResponse.accounts[0].accountId,
            // "source": userPrincipalsResponse.streamerInfo.appId,
            // "parameters": {
            //     "keys": "/CL",
            //     "fields": "0,1,2,3,4,5,6,7"


            // Historical Data
            "service": "CHART_HISTORY_FUTURES",
                        "requestid": "2",
                        "command": "GET",
                        "account": userPrincipalsResponse.accounts[0].accountId,
                        "source": userPrincipalsResponse.streamerInfo.appId,
                        "parameters": {
                            "symbol": "/CL",
                            "frequency": "m1",
                            "period": "y3"

            // If internet connection drops out
            // "service": "CHART_HISTORY_FUTURES",
            //             "requestid": "2",
            //             "command": "GET",
            //             "account": userPrincipalsResponse.accounts[0].accountId,
            //             "source": userPrincipalsResponse.streamerInfo.appId,
            //             "parameters": {
            //                 "symbol": "/CL",
            //                 "frequency": "m1",
            //                 "period": "y3"

						}
					}
			]
		}
		mySock.send(JSON.stringify(request));
	},1000)
},1000);

// (USE THIS) - Historical Data Data-folder: node CL_stream.js > /media/melgazar9/95d0b8d6-6582-4af9-8c3e-d9497417f0d1/Trading/Data/CL/historical-data/CL_1min/CL_1min_historical_data_$(date +"%Y-%m-%d-%I:%M:%p").log

// Live Stream: node CL_stream.js > ~/Trading/TD/Live-Trading/CL/live-stream-data/CL_1min_live-logs/CL_1min_$(date +"%Y-%m-%d-%I:%M:%p").log

// If connection Lost: node CL_stream.js > ~/Trading/TD/Live-Trading/CL/dropped_connection_1min_data/CL_1min_historical_dropped_connection_$(date +"%Y-%m-%d-%I:%M:%p").log
