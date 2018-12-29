---
layout: post
title: "How I setup this Blog"
date: 2018-12-28
---

This post describes how I setup my Blog using Jekyll and GitHub

## Setting up the GitHub part
- Went to Github.com &rarr; Created a new repository &rarr; `<username>.github.io`
- Cloned that repository on my computer using 
    ```
    git clone <repo address>
    ```

## Setting up the Jekyll part
### Installation
- Installed Ruby using ```sudo apt-get install ruby-full```
- Installed jekyll using ```gem install bundler jekyll```

### Creating the blog
- Went to the parent folder of the local git repo
- Scaffolded the basic jekyll blog structure
    ```
    jekyll new <username>.github.io
    ```
- Edited the `_config.yml` file for my details

## Checking Posts
Before committing my post to GitHub I local check it with `jekyll serve` and logging into http://localhost:4000
