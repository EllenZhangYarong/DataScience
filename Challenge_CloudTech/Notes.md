# March 5, 2020

## Lesson 12

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

## Lesson 13

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

## LESSON 17 Messaging & Containers

### Messaging in the Cloud
	There are often times that users of your applications need to be notified when certain events happen. Notifications, such as text messages or emails can be sent through services in the cloud. The use of the cloud offers benefits like lowered costs, increased storage, and flexibility.

**Messaging typically occurs between Internet-based applications and devices. One system can send a message to another system.**

### Simple Notification Service
	Amazon Simple Notification Service (or SNS) is a cloud service that allows you to send notifications to the users of your applications. SNS allows you to decouple the notification logic from being embedded in your applications and allows notifications to be published to a large number of subscribers.

#### Features
	SNS uses a publish/subscribe model.
	SNS can publish messages to Amazon SQS queues, AWS Lambda functions, and HTTP/S webhooks.

#### Tips
	SNS is found under the Application Integration section on the AWS Management Console.
	SNS Topic names are limited to 256 characters.
	A notification can contain only one message.

**Notifications can be sent to end users using mobile push, text messages, and email.**

### Queues
	A queue is a data structure that holds requests called messages. Messages in a queue are commonly processed in order, first in, first out (or FIFO).

	Messaging queues improve:
	- performance
	- scalability
	- user experience

**The use of asynchronous processing, where a user doesn't wait for a response, improves the overall user experience.**

### Simple Queue Service
	Amazon Simple Queue Service (SQS) is a fully managed message queuing service that allows you to integrate queuing functionality in your application. SQS offers two types of message queues: standard and FIFO.

#### Features
	- send messages
	- store messages
	- receive messages

#### Tips
	- The Simple Queue Service (SQS) is found under the Application Integration on the AWS Management Console.
	- FIFO queues support up to 300 messages per second.
	- FIFO queues guarantee the ordering of messages.
	- Standard queues offer best-effort ordering but no guarantees.
	- Standard queues deliver a message at least once, but occasionally more than one copy of a message is delivered.

**You should use FIFO ordering when message ordering is critical and standard queues when messages can arrive more than once and be processed out of order.**

### Containers in the Cloud
	Enterprises are adopting container technology at an explosive rate. A container consists of everything an application needs to run: the application itself and its dependencies (e.g. libraries, utilities, configuration files), all bundled into one package.

	Each container is an independent component that can run on its own and be moved from environment to environment.

**A container consists of everything an application needs to run: the application itself and its dependencies (e.g. libraries, utilities, configuration files), all bundled into one package.**

### Elastic Container Service (ECS)
	ECS is an orchestration service used for automating deployment, scaling, and managing of your containerized applications. ECS works well with Docker containers by:
	- launching and stopping Docker containers
	- scaling your applications
	- querying the state of your applications

### Tips
	- ECS is under the Compute section on the AWS Management Console.
	- You can schedule long-running applications, services, and batch processeses using ECS.
	- Docker is the only container platform supported by Amazon ECS.

**ECS is used for automating deployment, scaling and managing your containerized applications.**

## LESSON 18 AWS Management

### Cloud Trail
	Cloud Trail allows you to audit (or review) everything that occurs in your AWS account. Cloud Trail does this by recording all the AWS API calls occurring in your account and delivering a log file to you.

#### Features
	CloudTrail provides event history of your AWS account activity, including:

who has logged in
services that were accessed
actions performed
parameters for the actions
responses returned
This includes actions taken through the AWS Management Console, AWS SDKs, command line tools, and other AWS services.

### Tips
	- Cloud Trail is found under the Management & Governance section on the AWS Management Console.
	- CloudTrail shows results for the last 90 days.
	- You can create up to five trails in an AWS region.

**Cloud Trail allows you to audit (or review) everything that occurs in your AWS account.**
	- Set up alerts and alarms for certain activities
	- Log responses from AWS services
	- Track calls made using the SDK

### Cloud Watch
	Cloud Watch is a service that monitors resources and applications that run on AWS by collecting data in the form of logs, metrics, and events.

#### Features
	- There are several useful features:
	- Collect and track metrics
	- Collect and monitor log files
	- Set alarms and create triggers to run your AWS resources
	- React to changes in your AWS resources

#### Tips
	- CloudWatch is found under the Management & Governance section on the AWS Management Console.
	- Metrics are provided automatically for a number of AWS products and services.

**Cloud Watch can collect and track metrics, collect and monitor log files, and create triggers to run your AWS resources.**

### Infrastructure as Code
	Infrastructure as Code allows you to describe and provision all the infrastructure resources in your cloud environment. You can stand up servers, databases, runtime parameters, resources, etc. based on scripts that you write. Infrastructure as Code is a time-saving feature because it allows you to provision (or stand up) resources in a reproducible way.

### Cloud Formation
	AWS Cloud Formation allows you to model your entire infrastructure in a text file template allowing you to provision AWS resources based on the scripts you write.

#### Tips
	- Cloud Formation is found under the Management & Governance section on the AWS Management Console.
	- Cloud Formation templates are written using JSON or YAML.
	- You can still individually manage AWS resources that are part of a CloudFormation stack.

**Since your infrastructure is now code, you can check your scripts into version control.**


## LESSON 19 Getting Started with CloudFormation
Set up the necessary tools to get started with AWS CloudFormation and deploy your first server.

### Lesson 1: Introduction to Cloud Formation
### Lesson 2: Understanding Diagrams of Cloud Architecture
### Lesson 3: Infrastructure as Code (convert diagrams into code)
### Lesson 4: Deploying Services
### Lesson 5: Additional services that you’ll use in the project

Issues that DevOps tries to solve:
- Unpredictable deployments
- Mismatched environments (development doesn’t match production)
- Configuration Drift


	DevOps gives best practices and tools for solving these problems:
		- DevOps Tools: DevOp tools deploy and manage configuration changes to servers.
		Stackexchange has a discussion post detailing the difference between DevOps tools vs. Software Configuration Tools
		- Allows for predictable deployments, because it’s an automated script.
		- Enables Continuous Integration Continuous Deployment (CI/CD) so that new features are automatically deployed with all the required dependencies.

Glossary
- **Continuous Integration Continuous Deployment (CI/CD)**: Tracks the development workflow from testing through production. Continuous integration is process flow of testing any change made to your development flow, while continuous deployment tracks those changes through to staging and production systems.
- Check out this article by [Atlassian.com](https://www.atlassian.com/continuous-delivery/principles/continuous-integration-vs-delivery-vs-deployment) that describes these in detail.
- This [article](https://css-tricks.com/continuous-integration-continuous-deployment/) by Florian Matlik describes the differences between continuous integration and continuous deployment.

- **CloudFormation**: CloudFormation is a tool in AWS for managing, configuring and deploying infrastructure (push code along with the necessary server configurations).

*You'll need these three tools to get started in CloudFormation. Version Control,Code Editor for YAML and JSON,Amazon Web Services account*

### Creating Access Key ID

Deciding Access Privileges within AWS

	- Programmatic Access

	In the AWS console, choose "programmatic access." This allows us to use code to interact with AWS, instead of relying on mouse clicking in the console web pages.

	- Administrator Access

	For IAM access, choose “administrator access.” This is just for initial setup of your account. Afterwards, you’ll want to limit access to only what you need.

Dev and Prod user accounts

	In practice, Dev and DevOps members may have separate user accounts for the dev environment as opposed to the production environment. This makes it easier for developers by giving them wider privileges in the dev environment that would normally only be reserved for DevOps members in the production environment.

Access Key ID and Secret Access Keys

	Remember not to save these in your code or to check into any repositories. Keep these private to you.


**Specifying a region is a nice-to-have and not mandatory, but it does make your life easier when using region-specific services.**

### Configuring AWS CLI

**Configuring the AWS Command Line Interface (CLI)**
	- Download and install the AWS CLI tool.
	- In the terminal, type aws --version: this verifies that you have the AWS CLI tool.
	- To set up your AWS CLI, type aws configure in the terminal. Next when prompted for the AWS Access Key ID, paste in your Secret access key.
	- Region: Please use us-west-2, even if you’re living closer to another available region.

**It's great practice to change them every 90 days or sooner and also to just delete them or mark them inactive when not in use.**

**Verifying your Setup**
	- One way to check if your AWS CLI is set up properly is to try a command. You can try listing your S3 buckets:aws s3 ls . This will be blank if you have no S3 buckets. However, if you have no error message, then you’ve verified that your user has API access to communicate with AWS.
	- Note that each user can have up to 2 access keys at the same time.

**Additional Access Keys**
	Note that each user can have up to 2 access keys at the same time.

**Why Making Keys Inactive is a Better Choice**
	You may make your access key temporarily inactive rather than destroying it and creating a new one. This may be helpful if you want to stop an automated process that uses that key (for example, a CI/CD process).

### Understanding CloudFormation

*CloudFormation is a declarative language, not an imperative language.*

	- CloudFormation handles resource dependencies, so that you don’t have to specify which resource to start up before another. There are cases where you can specify that a resource depends on another resource, but ideally, you’ll let CloudFormation take care of dependencies.

	- VPC is the smallest unit of resource.

**Declarative languages**: These languages specify what you want, without requiring you to specify how to get it. An example of a popular declarative language is SQL.

**Imperative languages**: These languages use statements to change the state of the program.


### Getting Started With CloudFormation Script

*If you have a team of database, operations and networking experts, you would split your CloudFormation script into several files based on:* **Type of resource**


#### YAML and JSON
	YAML and JSON file formats are both supported in CloudFormation, but YAML is the industry preferred version that’s used for AWS and other cloud providers (Azure, Google Cloud Platform).

	An important note about YAML files: the whitespace indentation matters! We recommend that you use four white spaces for each indentation.

#### Glossary in CloudFormation scripts

**Name**: A name you want to give to the resource (does this have to be unique across all resource types?)
**Type**: Specifies the actual hardware resource that you’re deploying.
**Properties**: Specifies configuration options for your resource. Think of these as all the drop down menus and checkbox options that you would see in the AWS console if you were to request the resource manually.
**Stack**: A stack is a group of resources. These are the resources that you want to deploy, and that are specified in the YAML file.

#### Best practices
	Coding best practice: Create separate files to organize your code. You can either create separate files for similar resources, or create files for each developer who uses those resources.

### Testing CloudFormation

`aws cloudformation create-stack --stack-name myfirsttest --region us-west-2 --template-body file://uda_cloudformation_test.yml`


## LESSON 20 Infrastructure Diagrams

Convert business requirements into infrastructure diagrams and understand the principles behind design choices.

*Cloud Architecture diagrams are a way to visualize the infrastructure you want to design.*

*If you have too much going on inside a subnet, for example, you could create a Cloud Architecture diagram just for that one subnet and have a separate diagram that shows the VPC, AWS Account and/or region.*

*If my diagram covers multiple AWS accounts, regions and subnets, I should Create multiple diagrams, with each diagram covering a logical container, such as subnet, VPC or Account.*


*Availability Zones (AZ): An AZ is a data center (physical building).*

### Best Practices
	- Choose to have more than one availability zone to avoid a single point of failure.
	- Include more than one availability zone to design for high availability, .
	- You may choose to reduce to one AZ, possibly for prototyping and design for low cost. But it is not recommended for production environments.

**Virtual Private Cloud (VPC)**: A virtual private cloud is a pool of networked cloud resources. It can span more than one availability zone.

The equivalent of this would be a data center. However, thanks to availability zones, VPCs can span more than one physical building. This is an amazing feature that protects against real world disasters like electrical failures, fires and similar events.

### Subnets
	- A subnet is a subset of the overall VPC network and it only exists in a single availability zone, unlike its parent network, the VPC.

	- A subnet contains resources, and can be assigned access rights that apply to all resources within that subnet.

	- Subnets can be public or private. Public subnets are accessible to external users. Private subnets are only accessed internally by other resources within your cloud container.

####Use IP addresses for routing traffic
	- Use IP addresses as the “keys” for routing traffic. We can route traffic to stay within the VPC, or within a particular subnet, for security reasons.

	- For example, a database or any sensitive data will be placed in a private subnet. A public server, like a web server, can be placed in a public subnet. Routing rules applied to a subnet allow us to define access to all resources placed inside that subnet.


### Internet Gateway
	- An internet gateway is a resource that enables inbound and outbound traffic from the internet to your VPC.
	- An internet gateway allows external users access to communicate with parts of your VPC.
	- If you create a private VPC for an application that is internal to your company, you will not need an internet gateway.


*Network Address Translation (NAT) Gateway: provides outbound-only internet gateway for private services to access the internet. This keeps the private service protected from inbound connections, but allows it to connect to the internet in order to perform functions such as downloading software updates. The NAT gateway serves as an intermediary to take a private resource’s request, connect to the internet, and then relay the response back to the private resource without exposing that private resource’s IP address to the public.*

#### Note: 
Place NAT Gateways inside the public subnets and not the private subnets. NAT gateways need to be in the public subnet so that they can communicate with the public internet, and handle requests from resources that are in a private subnet.


	If I just created a VPC and I want to provide internet to it, I should make sure to
	- Create a route to the IGW and associate it with your subnet(s)
	- Create an IGW
	- Attach the IGW to your VPC

**consider these steps when troubleshooting a "no internet access" issue.**

### Autoscaling group: 
**An autoscaling group** manages multiple instances of the same resource (for example, servers), based on need. For instance, when there is a lot of internet traffic to a site, the autoscaling group can start more servers. When there is less traffic, the autoscaling group can reduce the number of servers.

#### Best Practice
	- It is recommended that an autoscaling group spans more than one availability zone, for reliability.
	- If we set the autoscaling group to run one resource, it will run that one resource in one of the availability zones.
	- If there is a failure of that resource, the autoscaling group will shut it down in that availability zone and start that same resource in the other availability zone.


### Load Balancer
	- A load balancer takes incoming traffic and distributes it to two or more resources. For example, it can take inbound user requests to access your website, and it can distribute the requests evenly among two or more servers.
	- Without a load balancer, having public-facing servers in more than one AZ would mean that users would have to use a different URL to reach each of the AZs. This can be impractical compared to just a single URL.

### Security Groups
	- Security groups manage traffic at the server level (the resource level). Security Groups aren’t for managing higher level groups such as subnets, VPC, or user accounts.
	- The same security group can be assigned to multiple resources that require the same security access settings defined by that security group.	

### S3
	- An S3 bucket is a public service for users to upload or download files.
	- Place the S3 service outside of your VPC.

**Images, Video, large text files, log files, audit logs are all great uses for S3.**
