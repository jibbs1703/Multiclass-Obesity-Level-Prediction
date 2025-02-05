import os
from dotenv import load_dotenv
from io import StringIO
import logging
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class S3Buckets:

    @classmethod
    def credentials(cls):
        """
        Retrieves AWS credentials from a hidden environment file.

        This class method accesses the user's AWS secret and access keys stored in an environment file.
        If a region is specified, the methods within the S3Buckets class will execute in that region.
        Otherwise, AWS will assign a default region.

        :param region: AWS region specified by the user (default is None)
        :return: An instance of the S3Buckets class initialized with the user's credentials and specified region
        """
        load_dotenv()
        secret = os.getenv("ACCESS_SECRET")
        access = os.getenv("ACCESS_KEY")
        region = os.getenv("REGION")

        return cls(secret, access, region)

    def __init__(self, secret, access, region):
        """
        Initializes the S3Buckets class with user credentials and creates the AWS S3 client.

        This constructor method initializes the S3Buckets class using the provided secret and access keys.
        It creates an AWS S3 client using the boto3 library. If no region is specified, AWS assigns a default
        region. The created client is available for subsequent methods within the class.

        :param secret: User's AWS secret key loaded from the environment file
        :param access: User's AWS access key loaded from the environment file
        :param region: Specified AWS region during instantiation (default is None)
        """
        if region is None:
            self.client = boto3.client("s3",
                                       aws_access_key_id=access,
                                       aws_secret_access_key=secret)
        else:
            self.location = {"LocationConstraint": region}
            self.client = boto3.client(
                "s3",
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                region_name=region,
            )

    def list_buckets(self):
        """
        Retrieves and returns a list of bucket names available in the user's AWS account.

        :return: A list of the S3 bucket instances present in the accessed account
        """
        response = self.client.list_buckets()
        buckets = response["Buckets"]
        all_buckets = [bucket["Name"] for bucket in buckets]
        logging.info(
            f"This account contains the following buckets: {all_buckets}")
        return all_buckets

    def create_bucket(self, bucket_name):
        """
        Creates an S3 bucket in the user's AWS account.

        This method creates a new S3 bucket in the region specified during the instantiation of the class.
        If the bucket name has already been used, it will not create a new bucket. If no region is specified,
        the bucket is created in the default S3 region (us-east-1).

        :param bucket_name: Name of the bucket to be created

        Returns: None
        """
        if bucket_name in self.list_buckets():
            logging.info(f"The bucket {bucket_name} already exists")
        else:
            logging.info("A new bucket will be created in your AWS account")
            self.client.create_bucket(Bucket=bucket_name,
                                      CreateBucketConfiguration=self.location)
            logging.info(
                f"The bucket {bucket_name} has been successfully created")

    def list_objects(self, bucket_name):
        """
        Lists the objects in an S3 bucket.

        :param bucket_name: Name of the S3 bucket.
        :return: List of object keys if successful, None otherwise.
        """
        try:
            response = self.client.list_objects_v2(Bucket=bucket_name)

            if 'Contents' in response:
                object_keys = [obj['Key'] for obj in response['Contents']]
                logging.info(
                    f"Objects in bucket '{bucket_name}': {object_keys}")
                return object_keys
            else:
                logging.info(f"No objects found in bucket '{bucket_name}'.")
                return []

        except ClientError as e:
            logging.error(f"Client Error listing objects: {e}")
            return None

        except Exception as e:
            logging.exception(f"Unexpected error listing objects: {e}")
            return None

    def upload_file(self, bucket_name, filename, file, folder=""):
        """
        Uploads a file to an S3 bucket.
        Parameters:
        - bucket_name (str): The name of the target S3 bucket.
        - filename (str): The name of the file to be uploaded.
        - file (io.BytesIO): The file object to be uploaded.
        - folder (str, optional): The folder path within the S3 bucket. Default is an empty string.

        Returns: None
        """
        try:
            self.client.put_object(Bucket=bucket_name,
                                   Key=f"{folder}{filename}",
                                   Body=file.getvalue())
            logging.info(
                f"File {filename} uploaded successfully to {bucket_name}/{folder}"
            )
        except ClientError as e:
            logging.error(f"Error uploading file to S3: {str(e)}")
        except Exception as e:
            logging.error(f"Error uploading file to S3: {str(e)}")

    def download_file(self, bucket_name, object_name, file_name) -> None:
        """
        Downloads a file from an S3 bucket in the user's AWS account.

        :param bucket_name: Name of the bucket to download the file from
        :param object_name: Name of the file to download from the S3 bucket
        :param file_name: Name of the file to save the downloaded content to
        :return: None
        """
        try:
            self.client.download_file(bucket_name, object_name, file_name)
            logging.info(
                f"File '{object_name}' downloaded successfully from bucket '{bucket_name}' to '{file_name}'."
            )

        except ClientError as e:
            logging.error(f"Client Error downloading file: {e}")

        except Exception as e:
            logging.exception(f"Unexpected error downloading file: {e}")

    def read_file(self, bucket_name, object_name):
        """
        Reads a file from an S3 bucket in the user's AWS account and returns its contents.

        :param bucket_name: Name of the bucket to read the file from.
        :param object_name: Name of the file to read from the S3 bucket.
        :return: An object containing the file's contents, or None if an error occurs.
        """
        try:
            response = self.client.list_objects_v2(Bucket=bucket_name)
            if 'Contents' not in response or not any(
                    obj['Key'] == object_name for obj in response['Contents']):
                logging.info(
                    f"The specified key '{object_name}' does not exist in bucket '{bucket_name}'."
                )
                return "File not in bucket"

            response = self.client.get_object(Bucket=bucket_name,
                                              Key=object_name)
            file_content = StringIO(response['Body'].read().decode('utf-8'))
            logging.info(
                f"File '{object_name}' read successfully from bucket '{bucket_name}'."
            )
            return file_content

        except ClientError as e:
            logging.error(f"Client Error reading file: {e}")
            return None

        except Exception as e:
            logging.exception(f"Unexpected error reading file: {e}")
            return None

    def delete_file(self, bucket_name, file_name):
        """
        Deletes a file from an S3 bucket in the user's AWS account.

        :param bucket_name: Name of the bucket to access the file.
        :param object_name: Name of the file to delete from the S3 bucket.
        :return: Message indicating the result of the deletion.
        """
        try:
            self.client.delete_object(Bucket=bucket_name, Key=file_name)
            logging.info(
                f"The file '{file_name}' has been deleted from the bucket '{bucket_name}'."
            )
            return f"The file '{file_name}' has been deleted from the bucket '{bucket_name}'."

        except ClientError as e:
            logging.error(f"Client Error deleting file: {e}")
            return f"Client Error: Unable to delete the file '{file_name}' from the bucket '{bucket_name}'."

        except Exception as e:
            logging.exception(f"Unexpected error deleting file: {e}")
            return f"Error: Unexpected error occurred while deleting the file '{file_name}' from the bucket '{bucket_name}'."
