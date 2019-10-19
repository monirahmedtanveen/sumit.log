---
layout: post
comments: false
title: "Send Cors Headers In Laravel Using Spatie Laravel Cors Package"
date: 2019-09-18 10:18:00
tags: laravel
---

>  This is an application which have laravel backend and jQuery frontend. In this application I solved the CORS(Cross Origin Resource Sharing) headers error while sharing data between frontend and backend. I used spatie/laravel-cors Package in backend app to solve the CORS headers issue.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
This is an application which have laravel backend and jQuery frontend. In this application I solved the CORS(Cross Origin Resource Sharing) headers error while sharing data between frontend and backend. I used <a href="https://github.com/spatie/laravel-cors" target="_blank">spatie/laravel-cors</a> Package in backend app to solve the CORS headers issue.

## Creating Backend Api
In this section I will create an api called <strong>users</strong> which will fetch all the user from the database. For creating this api at first install a freash laravel project into a folder called <strong>laravel-cors</strong>.

```php
composer create-project --prefer-dist laravel/laravel backend
```

After installation, configure the .env file. Change the following lines of .env

```php
...
DB_DATABASE=your_database
DB_USERNAME=your_database_username
DB_PASSWORD=your_database_password
```

Change the database/migrations/2014_10_12_000000_create_users_table.php migration file

```php
<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateUsersTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('users', function (Blueprint $table) {
            $table->increments('id');
            $table->string('name');
            $table->string('email')->unique();
            $table->string('password');
            $table->string('mobile')->nullable();
            $table->string('address')->nullable();
            $table->rememberToken();
            $table->integer('created_at');
            $table->integer('updated_at');
            $table->integer('deleted_at')->nullable();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('users');
    }
}
```

Before migration it is need to add bellow statement on boot method of <strong>app\Providers\AppServiceProvider</strong> class

```php
Schema::defaultStringLength(191);
```

Then migrate this table using following command

```php
php artisan migrate
```

It will create a <strong>users</strong> table in your database. Now it’s time for generating some dummy user data in users table. The seed code for creating dummy user placed in <strong>DatabaseSeeder.php</strong> file. This file is located in <strong>database\seeds</strong> folder. Bellow is the seed code of DatabaseSeeder file

```php
<?php

use App\User;
use Illuminate\Database\Seeder;
use Faker\Generator as Faker;

class DatabaseSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run(Faker $faker)
    {
        for ($i = 0; $i < 10; $i++) {
            User::create([
                'name' => $faker->name,
                'email' => $faker->unique()->safeEmail,
                'mobile' => $faker->phoneNumber,
                'address' => $faker->address,
                'password' => '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', // secret
                'remember_token' => str_random(10),
            ]);
        }
    }
}
```

Now seed the user table by running bellow command

```php
php artisan db:seed
```

Now after creating the database tables and seed some dummy users, let us create a responser trait. Create a file called <strong>ApiResponser.php</strong> in <strong>app\Traits</strong> folder. Bellow is the full code of <strong>ApiResponser.php</strong> file

```php
<?php

namespace App\Traits;

use Illuminate\Support\Collection;
use Illuminate\Database\Eloquent\Model;


trait ApiResponser {
    protected function showAll (Collection $collection, $code = 200) {
        return $this->successResponse(['data' => $collection], $code);
    }
}
```

After that create a controller named <strong>ApiController</strong> in <strong>app\Http\Controllers\Api\V100</strong> folder. This controller will extend the <strong>app\Http\Controllers\Controller</strong> class and use the <strong>app\Traits\ApiResponser</strong> trait. Below is the full code of ApiController class

```php
<?php

namespace App\Http\Controllers\Api\V100;

use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use App\Traits\ApiResponser;

class ApiController extends Controller
{
    use ApiResponser;
}
```

Now create a controller named <strong>UserController</strong> in <strong>app\Http\Controllers\Api\V100\user</strong> folder. <strong>UserController</strong> will extend the <strong>app\Http\Controllers\Api\V100\ApiController</strong> class. Next create index function inside the <strong>UserController</strong> which will fetch all the user data from the users table of database. Here is the code of <strong>UserController</strong>

```php
<?php

namespace App\Http\Controllers\Api\V100\user;

use App\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use App\Http\Controllers\Api\V100\ApiController;

class UserController extends ApiController
{
    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index()
    {
        $users = User::get();

        return $this->showAll($users);
    }
}
```

Now set up the api route prefix in <strong>app\Providers\RouteServiceProvider</strong> class. Here how the <strong>RouteServiceProvider</strong> class looks like

```php
<?php

namespace App\Providers;

use Illuminate\Support\Facades\Route;
use Illuminate\Foundation\Support\Providers\RouteServiceProvider as ServiceProvider;

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

Here in mapApiRoutes method I set the api route prefix as <strong>api/v1.0.0</strong>.

Now create create a api route named <strong>users</strong> in <strong>routes\api.php</strong> file.

```php
...
Route::resource("users", "user\UserController", ["only" => ["index"]]);
```

It is a GET api which will return all user data as json encoded form.

<img src="https://monirahmedtanveen.github.io/tech.logs/assets/images/posts/2019-09-18-send-cors-headers-in-laravel-using-spatie-laravel-cors-package/json_encoded_response.png" alt="json encoded response">

## Creating Frontend Website
In this section I will create website which will make a GET request to <strong>backend/api/v1.0.0/users</strong> api for getting the user data and list all the user data in a table.

At first create a <strong>index.html</strong> file inside the <strong>laravel-cors\fronend</strong> folder. Here is code of <strong>index.html</strong> file

```html
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8" />
	<meta content="width=device-width,
    initial-scale=1.0,maximum-scale=1.0"
    name="viewport">
	<meta property="og:title" content="You Title" />
	<meta property="og:type" content="website" />
	<meta property="og:image" content="Your-Image-Url" />
	<meta property="og:description" content="Your Description" />
	<title>Laravel Cors - User Data</title>

</head>
<body>
	<button onclick="loadData();" id="btnLoad">Load Data</button>
	<button onclick="clearData();" id="btnClear">Clear Data</button>
	<div id="data">
			click load data button to load user data
	</div>

	<div>
		<h3>User List</h3>
		<table border="1" style="text-align: center">
			<thead>
				<tr>
					<th>Name</th>
					<th>Email</th>
					<th>Mobile</th>
					<th>Address</th>
				</tr>
			</thead>

			<tbody id="tablebody">
				<tr id="rowmsg">
					<td colspan="4">No data to show</td>
				</tr>
			</tbody>
		</table>
	</div>

	<script src="js/jquery.min.js"></script>
	<script src="js/scripts.js"></script>
</body>
</html>
```

This page will look like this

<img src="https://monirahmedtanveen.github.io/tech.logs/assets/images/posts/2019-09-18-send-cors-headers-in-laravel-using-spatie-laravel-cors-package/webpage.png" alt="frontend website">

If I click on the <strong>Load Data</strong> button, it will triggered an event <strong>loadData()</strong> which is responsible for making GET request to <strong>backend/api/v1.0.0/users</strong>.

Now let us write the <strong>loadData()</strong> function. Create a <strong>script.js</strong> file inside the <strong>laravel-cors\fronend\js</strong> folder. Here is the code of <strong>script.js</strong> file. It will send a GET request to <strong>backend/api/v1.0.0/users</strong> api and list all the data to the table of <strong>index.html</strong> file.

```js
function loadData(){
	url = "http://localhost:8000/api/v1.0.0/users";
	$("#btnLoad").text("Loading Data");
	$("#btnLoad").attr({disabled: true}); 
		
	$.ajax({
		type: "GET",
		dataType: "json",
		
		url: url,
		data: {
			/*name: 'John Due',
			email: 'johndue@gmail.com',
			mobile: '01652564515',
			password: 'quioite',
			address: 'Dhaka-1300',*/
		},
		
		success: function(result,status,xhr){
			console.log("success");
			var data = JSON.parse(JSON.stringify(result));
			$('#rowmsg').hide();
			$.each(data['data'], function (index, value) {
				$("#tablebody").append(
					'<tr>' + 
						'<td>Name : ' + ' ' + value.name + '</td>'
						+ '<td>Email : ' + ' ' + value.email + '</td>'
						+ '<td>Mobile : ' + ' ' + value.mobile + '</td>'
						+ '<td>Address : ' + ' ' + value.address + '</td>'
					+ '</tr>'
				);
			});
		},
		error: function(xhr,status,error){
			console.log("error");
			$("#data").text(error);
		},
		complete: function(xhr,status){
			console.log("Complete");
			$("#btnLoad").text("Load Data");
			$("#btnLoad").attr({disabled: false});
		}
	});
}

function clearData(){
	$("#data").html(
		'click load data button to load user data'
	);
	$("#tablebody").html("<tr id='rowmsg'><td colspan='4'>No data to show</td></tr>");
}
```

## Frontend Request Blocked by CORS Policy
By following the above instruction I have created two different origing sharing data. One is backend which will give response of all user data and another is frontend which will receive the user data from the backend by making GET request. But in the browser CORS policy will block the fronend request as the backend has not specified which origins are permitted for sharing data with the backend. Here is the picture of CORS policy blocking fronend GET request

<img src="https://monirahmedtanveen.github.io/tech.logs/assets/images/posts/2019-09-18-send-cors-headers-in-laravel-using-spatie-laravel-cors-package/cors-block.png" alt="cors block">

When I click to the Load Button, instead of giving the data, the browser returns with CORS error. So I need to allow the frontend for sharing data with backend. I’m using <a href="https://github.com/spatie/laravel-cors" target="_blank">spatie/laravel-cors</a> Package to solve this issue.

## Installing <a href="https://github.com/spatie/laravel-cors" target="_blank">spatie/laravel-cors</a> Package
For installing <a href="https://github.com/spatie/laravel-cors" target="_blank">spatie/laravel-cors</a> Package, you have to run the following command into your laravel project.

```php
composer require spatie/laravel-cors
```

The package will automatically register its service provider. The provided <strong>Spatie\Cors\Cors</strong> middleware must be registered in the global middleware group.

In <strong>app/Http/Kernel.php</strong> register the middleware:

```php
protected $middleware = [
    ...
    \Spatie\Cors\Cors::class
];
```

After that publish the cors file by running bellow command

```php
php artisan vendor:publish --provider="Spatie\Cors\CorsServiceProvider" --tag="config"
```

Here is the default content of the config file published at <strong>config/cors.php</strong>:

```php
<?php

return [
    /*
     * A cors profile determines which orgins, methods, headers are allowed for
     * a given requests. The `DefaultProfile` reads its configuration from this
     * config file.
     *
     * You can easily create your own cors profile.
     * More info: https://github.com/spatie/laravel-cors/#creating-your-own-cors-profile
     */
    'cors_profile' => Spatie\Cors\CorsProfile\DefaultProfile::class,

    /*
     * This configuration is used by `DefaultProfile`.
     */
    'default_profile' => [

        'allow_credentials' => false,

        'allow_origins' => [
            '*',
        ],

        'allow_methods' => [
            'POST',
            'GET',
            'OPTIONS',
            'PUT',
            'PATCH',
            'DELETE',
        ],

        'allow_headers' => [
            'Content-Type',
            'X-Auth-Token',
            'Origin',
            'Authorization',
        ],

        'expose_headers' => [
            'Cache-Control',
            'Content-Language',
            'Content-Type',
            'Expires',
            'Last-Modified',
            'Pragma',
        ],

        'forbidden_response' => [
            'message' => 'Forbidden (cors).',
            'status' => 403,
        ],

        /*
         * Preflight request will respond with value for the max age header.
         */
        'max_age' => 60 * 60 * 24,
    ],
];
```

After installing this middleware the Backend API routes should now get appropriate CORS headers. The default configuration of this package allows all requests from any origin (denoted as ‘*’). If you want to specify some origins relevant to your project, lets say If you want to allow requests to come in from <strong>https://spatie.be</strong> and <strong>https://laravel.com</strong> add those domains to the config file:

```php
// config/cors.php

    ...
    'default_profile' => [

    'allow_origins' => [
        'https://spatie.be',
        'https://laravel.com',
    ],
    ...
...
```

## Working Demo
After successfully installing and configuring the <a href="https://github.com/spatie/laravel-cors" target="_blank">spatie/laravel-cors</a> Package, the Backend now allow the Frontend to have its data. Here is the picture of solving CORS header issue

<img src="https://monirahmedtanveen.github.io/tech.logs/assets/images/posts/2019-09-18-send-cors-headers-in-laravel-using-spatie-laravel-cors-package/working-demo.png" alt="working demo">

## Source Code
Click <a href="https://github.com/monirahmedtanveen/laravel-cors" target="_blank">here</a> to go the source code.

Thanks for reading. I hope you find this post useful.