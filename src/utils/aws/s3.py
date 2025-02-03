import os
from dotenv import load_dotenv
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd

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
            self.client = boto3.client(
                "s3", aws_access_key_id=access, aws_secret_access_key=secret
            )
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
        logging.info(f"This account contains the following buckets: {all_buckets}")
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
            self.client.create_bucket(
                Bucket=bucket_name, CreateBucketConfiguration=self.location
            )
            logging.info(f"The bucket {bucket_name} has been successfully created")

    def list_files(self, bucket_name, folder=""):
        """
        Lists files in an S3 bucket.
        Parameters:
        - bucket_name (str): The name of the S3 bucket.
        - folder (str, optional): The folder path within the S3 bucket. Default is an empty string.

        Returns: list: A list of filenames in the specified S3 bucket and folder.

        Logs:
        - Info: On successfully retrieving the list of files.
        - Error: If there's an error during the list retrieval."""

        try:
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=folder)
            if "Contents" in response:
                files = [item["Key"] for item in response["Contents"]]
                logging.info(
                    f"Files retrieved successfully from {bucket_name}/{folder}"
                )
                return files
            else:
                logging.info(f"No files found in {bucket_name}/{folder}")
                return []
        except ClientError as e:
            logging.error(f"Error retrieving file list from S3: {str(e)}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []

    def upload_file(self, bucket_name, filename, file, folder=""):
        """
        Uploads a file to an S3 bucket.
        Parameters:
        - bucket_name (str): The name of the target S3 bucket.
        - filename (str): The name of the file to be uploaded.
        - file (io.BytesIO): The file object to be uploaded.
        - folder (str, optional): The folder path within the S3 bucket. Default is an empty string.

        Returns: None

        Logs:
        - Info: On successful upload of the file.
        - Error: If there's an error during upload.
        - Error: If an unexpected error occurs.
        """
        try:
            self.client.put_object(
                Bucket=bucket_name, Key=f"{folder}{filename}", Body=file.getvalue()
            )
            logging.info(
                f"File {filename} uploaded successfully to {bucket_name}/{folder}"
            )
        except ClientError as e:
            logging.error(f"Error uploading file to S3: {str(e)}")
        except Exception as e:
            logging.error(f"Error uploading file to S3: {str(e)}")

    def download_file(self, bucket_name: str, s3_key: str, local_path: str) -> bool:
        """
        Downloads a file from an S3 bucket.

        Args:
            bucket_name: The name of the S3 bucket.
            s3_key: The S3 key (path) of the file.
            local_path: The local file path where the file will be downloaded. 
            local_path should include folder+filename.

        Returns:
            True on successful download, False otherwise.
        """
        try:
            # Ensure the local directory exists
            local_dir = os.path.dirname(local_path)
            if local_dir:  # Check if a directory part exists
                os.makedirs(local_dir, exist_ok=True)  # Create directory if needed

            self.client.download_file(bucket_name, s3_key, local_path)

            logging.info(f"File '{s3_key}' downloaded successfully to '{local_path}' from bucket '{bucket_name}'.")

        except ClientError as e:
            logging.error(f"Client Error downloading file: {e}")
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.error(f"The specified key '{s3_key}' does not exist in bucket '{bucket_name}'.")
            elif e.response['Error']['Code'] == 'NoSuchBucket':
                logging.error(f"The bucket '{bucket_name}' does not exist.")

        except Exception as e:
            logging.exception(f"Unexpected error downloading file: {e}")