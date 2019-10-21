---
layout: post
comments: false
title: "Converting Html To Pdf Using Barryvdh/laravel Snappy Package"
date: 2019-08-28 11:35:00
tags: laravel
---

>  Laravel Snappy package uses Wkhtmltopdf to convert html to pdf. The performance of this package is way better then some other pdf generating library which is wriiten in php like :- dompdf, mpdf, tcpdf etc.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Laravel Snappy package uses <a href="https://wkhtmltopdf.org/" target="_black">Wkhtmltopdf</a> to convert html to pdf. The performance of this package is way better then some other pdf generating library which is wriiten in php like :- dompdf, mpdf, tcpdf etc. As laravel snappy uses <a href="https://wkhtmltopdf.org/" target="_black">Wkhtmltopdf</a>, it can generate pdf of large amount of data withing a very short time. On the other hand, dompdf, mpdf and tcpdf fails dealing with large amount of data.

## Source Code
Click <a href="https://github.com/monirahmedtanveen/laravel-snappy" target="_blank">here</a> to go the source code.

Thanks for reading. I hope you find this post useful.