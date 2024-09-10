# Configuring Your Dockerfile

Before building your Docker image, you must set several environment variables that the container will use:

- `YOUR_USER_NAME`: Specify the username you wish to use within the container.
- `YOUR_GID`: Your group ID (GID) on the host machine. You can obtain this by running `id` in a bash terminal.
- `YOUR_UID`: Your user ID (UID) on the host machine. This can also be obtained by running `id` in bash.

These settings are necessary for the proper operation of `jax` and `timesfm` and I don't know why they need these.

# Building the Application

To build the application, ensure that the `dockerfile` is in the same directory as the `timesfm` folder. The directory structure should be as follows:

```
|
|--timesfm
|--dockerfile
```

Then, execute the following command in your terminal to build the Docker image:

```bash
docker build -t timesfm -f dockerfile .
```

# Selecting a Dockerfile Version

Choose the appropriate Dockerfile based on your source requirements:

- Use `dockerfile_mirror` if you need to build the Docker image from a mirror.
- Otherwise, use the standard `dockerfile` for other purposes.