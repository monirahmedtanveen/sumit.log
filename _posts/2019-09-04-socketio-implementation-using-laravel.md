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

### NodeJs & npm Installation
Command to install nodeJs & npm on linux environment:

```php
sudo apt-get install nodejs
sudo apt-get install npm
```

It will creates a class named SendEmailToUser in app\Console\Commands directory. Now edit the SendEmailToUser to create an artisan command.

For checking the version of nodeJs and npm run the following commands in terminal:

```php
nodejs -v
npm -v
```

### Laravel Project Dependencies
Go insode of your laravel project directory and install the following dependencies:

<strong><i>socket.io</i></strong>

```php
npm install socket.io --save
```

To check, if installed, type in:

```php
npm list socket.io
```

<strong><i>ioredis</i></strong>

```php
npm install ioredis --save
```

To check, if installed, type in:

```php
npm list ioredis
```

<strong><i>predis</i></strong>

```php
composer require predis/predis
```

## Setup Broadcasting Driver
Go insode of your laravel project directory and open .env file and setup the broadcasting driver.

```php
...
BROADCAST_DRIVER=redis
```

If you are using laravel 6.*, then you need to set the REDIS_CLIENT to predis in config/database.php. By default its set to phpredis.

```php
...
'client' => env('REDIS_CLIENT', 'predis'),
```

## Creating Larevel Event For Broadcasting Any Event
Inside of your laravel project run the following command in terminal to create an event called UserBroadcast(it can be anything).

```php
php artisan make:event UserBroadcast
```

The event class will be placed into app/Events folder. After creating the event you have to implement ShouldBroadcastNow in UserBroadcast.

```php
class UserBroadcast implements ShouldBroadcastNow
```

Here is how its look like

```php
<?php

namespace App\Events;

use App\User;
use Illuminate\Broadcasting\Channel;
use Illuminate\Queue\SerializesModels;
use Illuminate\Broadcasting\PrivateChannel;
use Illuminate\Broadcasting\PresenceChannel;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Contracts\Broadcasting\ShouldBroadcastNow;

class UserBroadcast implements ShouldBroadcastNow
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $user;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public function __construct(User $user)
    {
        $this->user = $user;
    }

    /**
     * Get the channels the event should broadcast on.
     *
     * @return \Illuminate\Broadcasting\Channel|array
     */
    public function broadcastOn()
    {
        return new Channel('user-channel');
    }
}
```

In this event I created a channel called <strong>user-channel</strong> in which data will be broadcasted.

## Creating Controller Class and Api Route to Fire the UserBroadcast Event
Run the following command in terminal to create a controller called UserController(it can be anything).

```php
php artisan make:controller Api/V100/UserController
```

Here is the full UserController

```php
<?php

namespace App\Http\Controllers\Api\V100\user;

use App\User;
use Illuminate\Http\Request;
use App\Events\UserBroadcast;
use App\Http\Controllers\Controller;

class UserController extends Controller
{
    /**
     * Store a newly created resource in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\Response
     */
    public function store(Request $request)
    {
        $rules = [
            'name' => 'required',
            'email' => 'required|email',
        ];
        $this->validate($request, $rules);

        $user = User::create([
            'name' => $request->name,
            'email' => $request->email,
            'password' => bcrypt('qwerty'),
        ]);

        /** Created User Broadcast Event */
        try {
            event(new UserBroadcast($user));
        } catch (\Exception $e) {
            /** Do Nothing */
        }

        return ["data" => $user];
    }
}
```

Next, open the <strong>routes/api.php</strong> file and create a route.

```php
...
Route::resource("users", "user\UserController");
```

After that a change need to be made in <strong>app\Providers\RouteServiceProvider</strong> class. Here is the full <strong>RouteServiceProvider</strong> class.

```php
<?php

namespace App\Providers;

use Illuminate\Foundation\Support\Providers\RouteServiceProvider as ServiceProvider;
use Illuminate\Support\Facades\Route;

class RouteServiceProvider extends ServiceProvider
{
    /**
     * This namespace is applied to your controller routes.
     *
     * In addition, it is set as the URL generator's root namespace.
     *
     * @var string
     */
    protected $namespace = 'App\Http\Controllers';
    protected $namespace_v1 = 'App\Http\Controllers\Api\V100';

    /**
     * Define your route model bindings, pattern filters, etc.
     *
     * @return void
     */
    public function boot()
    {
        //

        parent::boot();
    }

    /**
     * Define the routes for the application.
     *
     * @return void
     */
    public function map()
    {
        $this->mapApiRoutes();

        $this->mapWebRoutes();

        //
    }

    /**
     * Define the "web" routes for the application.
     *
     * These routes all receive session state, CSRF protection, etc.
     *
     * @return void
     */
    protected function mapWebRoutes()
    {
        Route::middleware('web')
             ->namespace($this->namespace)
             ->group(base_path('routes/web.php'));
    }

    /**
     * Define the "api" routes for the application.
     *
     * These routes are typically stateless.
     *
     * @return void
     */
    protected function mapApiRoutes()
    {
        Route::prefix('api/v1.0.0')
             ->middleware('api')
             ->namespace($this->namespace_v1)
             ->group(base_path('routes/api.php'));
    }
}
```

## Creating Node Server For Subscriving The ‘user-channel’ & Emitting The Data Received From ‘user-channel’
Create file <strong>socket.js</strong> in your root directory of laravel project. Here is the sample Code of <strong>socket.js</strong> file:

```js
var server = require('http').Server();
var io = require('socket.io')(server);
var Redis = require('ioredis');
var redis = new Redis();
 
//redis.subscribe('user-channel'); // single channel
redis.psubscribe('*'); //multiple channels
 
//redis.on('message', function(channel, message) { // single channel
redis.on('pmessage', function(subscribed, channel, message) { //multiple channels
    message = JSON.parse(message);
    io.emit(channel + ':' + message.event, message.data);
    console.log(channel + ':' + message.event, message.data);
});
 
server.listen(3000, function () {
    console.log('Listening on Port 3000');
});
```

<strong>Note:</strong> Above server has to be on, for sockets to work. For development, you may just use command in console inside of project root directory:

```php
node socket.js
```

## Client Side Code For Listening and Processing the Broadcast Data
I use laravel root link for client side webpage for listening and proccessing the broadcasted data.

```php
<!DOCTYPE html>
<html lang="">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>User List</title>

        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css?family=Nunito:200,600" rel="stylesheet">

        <!-- Styles -->
        <style>
            html, body {
                background-color: #fff;
                color: #636b6f;
                font-family: 'Nunito', sans-serif;
                font-weight: 200;
                height: 100vh;
                margin: 0;
            }

            .full-height {
                height: 100vh;
            }

            .flex-center {
                align-items: center;
                display: flex;
                justify-content: center;
            }

            .position-ref {
                position: relative;
            }

            .top-right {
                position: absolute;
                right: 10px;
                top: 18px;
            }

            .content {
                text-align: center;
            }

            .title {
                font-size: 40px;
            }

            .links > a {
                color: #636b6f;
                padding: 0 25px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: .1rem;
                text-decoration: none;
                text-transform: uppercase;
            }

            .m-b-md {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="flex-center position-ref full-height">
            <div class="content">
                <div class="title m-b-md">
                    User List
                </div>

                <div class="links">
                    <table border="1">
                        <thead>
                            <tr>
                                <th>Sl#</th>
                                <th>Name</th>
                                <th>Email Address</th>
                                <th>Created Date</th>
                            </tr>
                        </thead>
                        <tbody id="data">
                            @foreach ($users as $user)
                                <tr>
                                    <td></td>
                                    <td></td>
                                    <td></td>
                                    <td></td>
                                </tr>
                            @endforeach
                        <tbody>
                        <tfoot>
                            <tr>
                                <th>Sl#</th>
                                <th>Name</th>
                                <th>Email Address</th>
                                <th>Created Date</th>
                            </tr>
                        </tfoot>
                    </table>
                </div>
            </div>
        </div>

        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/1.7.2/socket.io.min.js"></script>
        <script>
        // this is all it takes to capture it in jQuery
        // you put ready-snippet
        $(function() {
            //you define socket - you can use IP
            var socket = io('http://localhost:3000');
            //you capture message data
            socket.on('laravel_database_user-channel:App\\Events\\UserBroadcast', function(data){
                //you append that data to DOM, so user can see it
                $('#data').append('<tr>'
                    + '<td>' + data.user.id + '</td>' 
                    + '<td>' + data.user.name + '</td>'
                    + '<td>' + data.user.email + '</td>'
                    + '<td>' + data.user.created_at + '</td>'
                    + '</tr>')
                // console.log(data.user.name);
            });
        });
        
        </script>
    </body>
</html>
```

## Working Demo
Api link for fire UserBroadcast event :- <strong>your-laravel-project-root/api/v1.0.0/users</strong>

Front End link for cathing the broadcasted data from UserBroadcast event :- <strong>your-laravel-project-root/</strong>

Go to your laravel project root link on a browser (<strong>link:- your-laravel-project-root/</strong>).

Then from Postman make a post request to <strong>your-laravel-project-root/api/v1.0.0/users<strong> with data :-

```php
[“name” : “jonh due(demo name)”, “email” : “john@gmail.com(demo email address)”]
```

<img src="https://monirahmedtanveen.github.io/tech.logs/assets/images/posts/2019-09-18-send-cors-headers-in-laravel-using-spatie-laravel-cors-package/demo.png" alt="working demo">

You will see the posted user data will update on browser’s user list real time.

## Source Code
Click <a href="https://github.com/monirahmedtanveen/laravel-socketio-implementation
" target="_blank">here</a> to go the source code.

Thanks for reading. I hope you find this post useful.