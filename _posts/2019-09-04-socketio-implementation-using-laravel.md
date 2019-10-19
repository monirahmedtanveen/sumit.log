---
layout: post
comments: false
title: "Socketio Implementation Using Laravel"
date: 2019-09-04 10:18:00
tags: laravel
---

>  Websockets is a web technology, which allows bi-directional, real-time communications between web client and a server. As  part of the HTML5 specification it works with the newest web browsers (including Internet Explorer 10 and its latest versions).


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Websockets is a web technology, which allows bi-directional, real-time communications between web client and a server. As  part of the HTML5 specification it works with the newest web browsers (including Internet Explorer 10 and its latest versions).

Socket.IO is a JavaScript library that helps improving work with WebSockets. It consists of two parts – server part (for Node.JS) and clients part (for web browsers). Both of them have similiar APIs based on event-driven architecture. Socket.IO allows to use additional features such as sending data to large number of sockets at the same time (broadcasting) or storing the data.

The main idea behind Socket.IO is the ability to send and receive any events with any data. It can be any object as well as a binary data.

## Prequisite
For successfull implementation of socketio some backend application need to be run. For this some dependencies need to be installed in your machine. Here are the list of dependencies and their installation process given.

<ul>
    <li>Redis Server</li>
    <li>NodeJs & npm</li>
</ul>

### Redis Server Installation
For Windows Environment, go to this link :- <a href="https://github.com/microsoftarchive/redis/releases" target="_blank">redis server</a> and download <strong>Redis-x64-3.0.504.zip</strong> file. After download unzip it and run the <strong>redis-server.exe</strong>.

For Linux Environment, command to install redis server :-

Update the apt-get packages :-

```php
sudo apt-get update
```

Next run below command from the terminal to install Redis on your machine :-

```php
sudo apt-get install redis-server
```

Next is to enable Redis to start on system boot. Also restart Redis service once.

```php
sudo systemctl enable redis-server.service
```

#### Install Redis PHP Extension 
If you need to use Redis from PHP application, you also need to install Redis PHP extension on your Ubuntu system. In our case it is mandatory. Run below command to install:

```php
sudo apt-get install php-redis
```

#### Test Connection to Redis Server
type below command and see the output,

```php
command :- "redis-cli"
output :- 127.0.0.1:6379>
command :- "ping"
output :- PONG (It output PONG if connection successful)
```

## Redis Server Installation
Go to your laravel projects root directory and run the folowing command to create an artisan command.

```php
php artisan make:command SendEmailToUser
```

It will creates a class named SendEmailToUser in app\Console\Commands directory. Now edit the SendEmailToUser to create an artisan command.

```php
/**
* The name and signature of the console command.
*
* @var string
*/
protected $signature = 'send:email';

/**
* The console command description.
*
* @var string
*/
protected $description = 'Sending a notification email about task to all users';

Now you have to register the command in app\Console\Kernel.php file.

/**
* The Artisan commands provided by your application.
*
* @var array
*/
protected $commands = [
   ...
   'App\Console\Commands\SendEmailToUser',
];
```

After that define the command schedule in schedule function of app\Console\Kernel.php file.

```php
protected function schedule(Schedule $schedule)
{
    $schedule->command('send:email')
        ->everyMinute();    /** Run the task every minute */
}
```

Now, if you run the php artisan list command in the terminal, you will see your command has been registered. You will be able to see the command name with the signature and description.

<img src="https://monirahmedtanveen.github.io/tech.logs/assets/images/posts/2019-09-28-scheduling-task-with-cron-job-in-laravel/send-email-command.png" alt="send:email command">

As the command is registered, I have wrriten my functionality in handle function of app\Console\Commands\SendEmailToUser class. If I run the send:email command, it will execute all the staments written in handle function of SendEmailToUser class. This is how the SendEmailToUser.php file looks with the handle method and all other changes in place:

```php
<?php

namespace App\Console\Commands;

use App\Mail\SendUserNotification;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Mail;
use App\User;

class SendEmailToUser extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'send:email';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Sending a notification email about task to all users';

    /**
     * Create a new command instance.
     *
     * @return void
     */
    public function __construct()
    {
        parent::__construct();
    }

    /**
     * Execute the console command.
     *
     * @return mixed
     */
    public function handle()
    {
        /** Sending Email */
        $users = User::limit(5)->get();

        foreach ($users as $user) {
            retry(5, function () use ($user) {
                Mail::to($user)->send(new SendUserNotification($user));
            }, 100);
        }

        $this->info('Notification Email Sent to All Users');
    }
}
```

If you run the following command

```php
php artisan send:email
```

You will see the email has been sent to all user and the output shows in the terminal.

<img src="https://monirahmedtanveen.github.io/tech.logs/assets/images/posts/2019-09-28-scheduling-task-with-cron-job-in-laravel/email-notification-send.png" alt="email notification sent">

## Starting the Laravel Scheduler
Let’s setup the Cron Jobs to run automatically without initiating manually by running the command. To start the Laravel Scheduler itself, we only need to add one Cron job which executes every minute. Go to your terminal, ssh into your server, cd into your project and run this command.

```php
crontab -e
```

This will open the server Crontab file, paste the code below into the file, save and then exit.

```php
* * * * * cd /path-to-your-project && php artisan schedule:run >> /dev/null 2>&1
```

Replace the /path-to-your-project with the full path to the Artisan command of your Laravel Application.

## Source Code
Click <a href="https://github.com/monirahmedtanveen/laravel-cron-job" target="_blank">here</a> to go the source code.

Thanks for reading. I hope you find this post useful.