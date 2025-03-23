# Data Intensive Applications

Any of the following

 Any of the following generation/usage increases quickly:
- Volume of data
- Complexity of data
- Speed of change in data

## Pillars

| Pillar       | Properties                                                                                                                                                                                                                                                                                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reliable     | Fault-tolerance<br>No authorized access<br>Chaos Testing<br>Robust to Full Machine Failures<br>Bug-free, Automated bug tests<br>Environments: Dev, Staging/Testing, Prod<br>Quick roll-backs                                                                                                                                                                    |
| Scalable     | Handle high traffic volume<br>Traffic load with peak # of reads, writes, simultaneous users<br>Capacity planning<br>Response time vs throughput<br>End user response time<br>90th, 95th percentile SLO/A service level objectives/agreements<br>Vertically-Scaling up (more powerful machine)<br>Horizontally-Scaling out (distributed across smaller machines) |
| Maintainable | Add new people to work<br>Productivity<br>Operable: Configurable and testable<br>Simple: easy to understand and ramp up, well-documented<br>Evolveable: easy to change                                                                                                                                                                                          |

## Components

![](assets/Data_Intensive_Applications_Typical_Components.png)


|                   |                                           | Tools                         |
| ----------------- | ----------------------------------------- | ----------------------------- |
| Databases         | Source of truth                           | SQL                           |
| Cache             | Temporary storage of expensive operation  | Memcache                      |
| Full-text index   | Quickly searching data by keyword/filter  | ESIndex<br>Apache Lucener     |
| Message queues    | MEssaging passing passing between process | Apache Kafka                  |
| Stream Processing |                                           | Apache Spark<br>Apache Samza  |
| Batch Processing  | Crunching last amount of data             | Apache Spark<br>Apache Hadoop |
| Application code  | Connective tissue other components        |                               |



