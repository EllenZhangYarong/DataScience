Shell Workshop

what's a shell?

**A shell is simply the outermost layer of an operating system. It's designed to provide a way for you to interact with the tools and services that your operating system provides.**

use your computer's text shell is called a *Command-Line Interface* or ***CLI***.


# Lesson 12

**Types of Cloud Computing**

*Infrastructure-as-a-Service (IaaS)*
	The provider supplies virtual server instances, storage, and mechanisms for you to manage servers.
*Platform-as-a-Service (PaaS)*
	A platform of development tools hosted on a provider's infrastructure.
*Software-as-a-Service (SaaS)*
	A software application that runs over the Internet and is managed by the service provider.

Resources:
[Types of Cloud Computing](https://aws.amazon.com/types-of-cloud-computing/)

**Cloud Deployment Models**

*Public Cloud*
	A public cloud makes resources available over the Internet to the general public. Resources could include servers, databases, application development services. AWS is currently the largest Public Cloud provider.
*Private Cloud*
	A private cloud is a proprietary network that supplies services to a limited number of people. Called On-premises.
*Hybrid Cloud*
	A hybrid model contains a combination of both a public and a private cloud. PII(Personally Identifiable Information) about customers maybe stored in a On-premise database for security reasons, while a web application to manage that data maybe served.

	The hybrid model is a growing trend in the industry for those organizations that have been slow to adopt the cloud due to being in a heavily regulated industry. The hybrid model gives organizations the flexibility to slowly migrate to the cloud.

# Lesson 13

### Elastic Cloud Compute

	Elastic Cloud Compute or EC2 is a foundational piece of AWS' cloud computing platform and is a service that provides servers for rent in the cloud.

	- On Demand - Pay as you go, no contract.
	- Dedicated Hosts - You have your own dedicated hardware and don't share it with others.
	- Spot - You place a bid on an instance price. If there is extra capacity that falls below your bid, an EC2 instance is provisioned. If the price goes above your bid while the instance is running, the instance is terminated.
	- Reserved Instances - You earn huge discounts if you pay up front and sign a 1-year or 3-year contract.

### Elastic Block Store
	Elastic Block Store (EBS) is a storage solution for EC2 instances and is a physical hard drive that is attached to the EC2 instance to increase storage.

### Virtual Private Cloud (VPC)
	Virtual Private Cloud or VPC allows you to create your own private network in the cloud. You can launch services, like EC2, inside of that private network. A VPC spans all the Availability Zones in the region.

	VPC allows you to control your virtual networking environment, which includes:

		- IP address ranges
		- subnets
		- route tables
		- network gateways

	Tips
		- VPC is found under Networking & Content Delivery section of the AWS Management Console.
		- The default limit is 5 VPCs per Region. You can request an increase for these limits.
		- Your AWS resources are automatically provisioned in a default VPC.
		- There are no additional charges for creating and using the VPC.
		- You can store data in Amazon S3 and restrict access so that it’s only accessible from instances in your VPC.

		EC2 Instances can be launched in a VPC, and you can store data in Amazon S3 and restrict access so that it’s only accessible from instances in your VPC.

*Compute power in the cloud is a faster way to build applications, providing:*

- no servers to manage (i.e. serverless)
- ability to continuously scale
- ability to run code on demand in response to events
- pay only when your code runs

### Lambda

AWS Lambda provides you with computing power in the cloud by allowing you to execute code without standing up or managing servers.


	Tips
	- Lambda is found under the Compute section on the AWS Management Console.
	- Lambdas have a time limit of 15 minutes.
	- The code you run on AWS Lambda is called a “Lambda function.”
	- Lambda code can be triggered by other AWS services.
	- AWS Lambda supports Java, Go, PowerShell, Node.js, C#/.NET, Python, and Ruby. There is a Runtime API that allows you to use other programming languages to author your functions.
	- Lambda code can be authored via the console.

	Lambda is one serverless technology offered by AWS.

	*What is the easiest way to author a Lambda function?*
		Lambda console editor in the AWS Management Console

	Lambda is event-driven, so you can run your code based on certain events happening, like a file upload, or a record being inserted in a database, etc.

### Elastic Beanstalk

Elastic Beanstalks is an orchestration service that allows you to deploy a web application at the touch of a button by spinning up (or provisioning) all of the services that you need to run your application.

	Tips
	- Elastic Beanstalk is found under the Compute section of the AWS Management Console.
	- Elastic Beanstalk can be used to deployed web applications developed with Java, .NET, PHP, Node.js, Python, Ruby, Go, and Docker.
	- You can run your applications in a VPC.

**Elastic Beanstalk can spin up database instances for you, VPCs, security groups, EC2 instances, etc.**


## LESSON 14
Storage & Content Delivery

### Storage in the Cloud

	Storage and database services in the cloud provide a place for companies to collect, store, and analyze the data they've collected over the years at a massive scale.

### Storage & Database Services

	- Amazon Simple Storage Service (Amazon S3)
	- Amazon Simple Storage Service (Amazon S3) Glacier
	- DynamoDB
	- Relational Database Service (RDS)
	- Redshift
	- ElastiCache
	- Neptune
	- Amazon DocumentDB

#### S3 & S3 Glacier
	Amazon Simple Storage Service (or S3) is an object storage system in the cloud.

##### Storage Classes
	S3 offers several storage classes, which are different data access levels for your data at certain price points.
	- S3 Standard
	- S3 Glacier
	- S3 Glacier Deep Archive
	- S3 Intelligent-Tiering
	- S3 Standard Infrequent Access
	- S3 One Zone-Infrequent Access
##### Tips
	- S3 is found under the Storage section on the AWS Management Console.
	- A single object can be up to 5 terabytes in size.
	- You can enable Multi-Factor Authentication (MFA) Delete on an S3 bucket to prevent accidental deletions.
	- S3 Acceleration can be used to enable fast, easy, and secure transfers of files over long distances between your data source and your S3 bucket.

#### DynamoDB
	DynamoDB is a NoSQL document database service that is fully managed. Unlike traditional databases, NoSQL databases, are schema-less. Schema-less simply means that the database doesn't contain a fixed (or rigid) data structure.

**Data is stored in DynamoDB in JSON or JSON-like format.**

##### Tips
	- DynamoDB is found under the Database section on the AWS Management Console.
	- DynamoDB can handle more than 10 trillion requests per day.
	- DynamoDB is serverless as there are no servers to provision, patch, or manage.
	- DynamoDB supports key-value and document data models.
	- DynamoDB synchronously replicates data across three AZs in an AWS Region.
	- DynamoDB supports GET/PUT operations using a primary key.	

**AWS is responsible for patching or upgrading the database. They are also responsible for provisioning or managing servers.**

### Relational Database Service (RDS)
	RDS (or Relational Database Service) is a service that aids in the administration and management of databases. RDS assists with database administrative tasks that include upgrades, patching, installs, backups, monitoring, performance checks, security, etc.

#### Database Engine Support
	- Oracle
	- PostgreSQL
	- MySQL
	- MariaDB
	- SQL Server

**RDS is able to manage popular database engines like Aurora, Oracle, PostgreSQL, MySQL, MariaDB, and SQL Server.**

**To deliver a managed service experience, Amazon RDS doesn't provide shell access to DB instances.**

### Redshift
	Redshift is a cloud data warehousing service to help companies manage big data. Redshift allows you to run fast queries against your data using SQL, ETL, and BI tools. Redshift stores data in a column format to aid in fast querying.

#### Tips
	- Redshift can be found under the Database section on the AWS Management Console.
	- Redshift delivers great performance by using machine learning.
	- Redshift Spectrum is a feature that enables you to run queries against data in Amazon S3.
	- Redshift encrypts and keeps your data secure in transit and at rest.
	- Redshift clusters can be isolated using Amazon Virtual Private Cloud (VPC).

** A data warehouse is used for reporting and data analysis.**

### Content Delivery In The Cloud
	A Content Delivery Network (or CDN) speeds up delivery of your static and dynamic web content by caching content in an Edge Location close to your user base.

#### Benefits
	The benefits of a CDN includes:
	- low latency
	- decreased server load
	- better user experience

### Cloud Front
	CloudFront is used as a global content delivery network (CDN). Cloud Front speeds up the delivery of your content through Amazon's worldwide network of mini-data centers called Edge Locations.

	CloudFront works with other AWS services, as shown below, as an origin source for your application:

	- Amazon S3
	- Elastic Load Balancing
	- Amazon EC2
	- Lambda@Edge
	- AWS Shield

## LESSON 14
Security In The Cloud

	As adoption of cloud services has increased, so has the need for increased security in the cloud. The great thing about cloud security is that it not only protects data, it also protects applications that access the data. Cloud security even protects the infrastructure (like servers) that applications run on.


	The way security is delivered depends on the cloud provider you're using and the cloud security options they offer.

### AWS Shield
	AWS Shield is a managed DDoS (or Distributed Denial of Service) protection service that safeguards web applications running on AWS.

	AWS Shield is a service that you get "out of the box", it is always running (automatically) and is a part of the free standard tier. If you want to use some of the more advanced features, you'll have to utilize the paid tier.

#### Tips
	- AWS Shield can be found under the Security, Identity, & Compliance section on the AWS Management Console.
	- AWS Shield Standard is always-on, using techniques to detect malicious traffic.
	- AWS Shield Advanced provides enhanced detection.


#### AWS Shield is a managed DDoS (or Distributed Denial of Service) protection service that safeguards web applications running on AWS.


**A Distributed Denial of Service (DDoS) attack is an attempt to make a website or an application unavailable by overwhelming it with traffic from multiple sources.**


### AWS WAF
	AWS WAF (or AWS Web Application Firewall) provides a firewall that protects your web applications.

	WAF can stop common web attacks by reviewing the data being sent to your application and stopping well-known attacks.

#### Tips
	- WAF is found under the Security, Identity, & Compliance section on the AWS Management Console.
	- WAF can protect web sites not hosted in AWS through Cloud Front.
	- You can configure CloudFront to present a custom error page when requests are blocked.

**AWS WAF helps protects your website from common attack techniques like SQL injection and Cross-Site Scripting (XSS).**

### Identity & Access Management
Identity & Access Management (IAM) is an AWS service that allows us to configure who can access our AWS account, services, or even applications running in our account. IAM is a global service and is automatically available across ALL regions.

#### Security Concepts
	- User : A person or service that interacts with services or applications running in your AWS account.
	- IAM Group: A collection of users.
	- IAM Role: Identity with permissions or a set of privileges.
	- Policy: Defines granular level permissions.

**You can create policies in JSON using the visual editor or the JSON editor in the IAM console.**

**MFA requires an additional form of authentication along with your password.**

##  LESSON 16
Networking & Elasticity

### Networking
	Networks reliably carry loads of data around the globe allowing for the delivery of content and applications with high availability. The network is the foundation of your infrastructure.

Cloud networking includes:
	- network architecture
	- network connectivity
	- application delivery
	- global performance
	- delivery

**The IP address is a long string of numbers that represents a computer's location on the Internet.**

**Domain name -> DNS -> Domain authority -> registration service -> Routed to address -> website displays**


### Route 53
	Route 53 is a cloud domain name system (DNS) service that has servers distributed around the globe used to translates human-readable names like www.google.com into the numeric IP addresses like 74.125.21.147.

#### Features
	- scales automatically to manage spikes in DNS queries
	- allows you to register a domain name (or manage an existing)
	- routes internet traffic to the resources for your domain
	- checks the health of your resources

#### Tips
	- Route 53 is found under the Networking & Content Delivery section on the AWS Management Console.
	- Route 53 allows you to route users based on the user’s geographic location.

**Route 53 does not provide website hosting as it is only an authoritative DNS service.**


### Elasticity in the Cloud
	One of the main benefits of the cloud is that it allows you to stop guessing about capacity when you need to run your applications. Sometimes you buy too much or you don't buy enough to support the running of your applications.

	With elasticity, your servers, databases, and application resources can automatically scale up or scale down based on load.

**Resources can scale up (or vertically). In Amazon EC2, this can easily be achieved by stopping an instance and resizing it to an instance type that has more RAM, CPU, IO, or you can scale out (or horizontally), which increases the number of resources. An example would be adding more servers.**

### EC2 Auto Scaling
	EC2 Auto Scaling is a service that monitors your EC2 instances and automatically adjusts by adding or removing EC2 instances based on conditions you define in order to maintain application availability and provide peak performance to your users.

#### Features
	- Automatically scale in and out based on needs.
	- Included automatically with Amazon EC2.
	- Automate how your Amazon EC2 instances are managed.

#### Tips
	- EC2 Auto Scaling is found on the EC2 Dashboard.
	- EC2 Auto Scaling adds instances only when needed, optimizing cost savings.
	- EC2 predictive scaling removes the need for manual adjustment of auto scaling parameters over time.

**You can configure EC2 Auto Scaling to send an SNS notification whenever your EC2 Auto Scaling group scales.**

### Elastic Load Balancing
	Elastic Load Balancing automatically distributes incoming application traffic across multiple servers.

	Elastic Load Balancer is a service that:
	- Balances load between two or more servers
	- Stands in front of a web server
	- Provides redundancy and performance

### Tips
	- Elastic Load Balancing can be found on the EC2 Dashbaoard.
	- Elastic Load Balancing works with EC2 Instances, containers, IP addresses, and Lambda functions.
	- You can configure Amazon EC2 instances to only accept traffic from a load balancer.

