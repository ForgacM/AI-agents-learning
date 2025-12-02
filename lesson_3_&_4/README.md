# Lesson 3
No code platform n8n

## Files
- agent_flow.json each LLM call have only one tool and whole flow is controlled manually
- AIAgent.json same flow but everything is inside AI_AGENT block

## Lessons learned
- Hard to debug tool calls input parameters
- lot of calls are hard to maintain it will be faster through code
- AI agent in place easy to use


## How to run 

```
docker volume create n8n_data

docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n 
```
Mock data for DB
```
CREATE TABLE weather_data (
                              id INT AUTO_INCREMENT PRIMARY KEY,
                              time DATETIME NOT NULL,
                              interval_seconds INT NOT NULL,
                              temperature_2m DOUBLE,
                              relative_humidity_2m INT,
                              apparent_temperature DOUBLE,
                              precipitation DOUBLE,
                              weather_code INT,
                              latitude DOUBLE,
                              longitude DOUBLE
);

INSERT INTO weather_data (
    time,
    interval_seconds,
    temperature_2m,
    relative_humidity_2m,
    apparent_temperature,
    precipitation,
    weather_code,
    latitude,
    longitude
) VALUES (
             '2025-12-02 18:15:00',
             900,
             1.9,
             95,
             -2.1,
             0.0,
             3,
             48.6287,
             17.4344
         );

```