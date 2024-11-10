import grpc
import recognition_report_pb2
import recognition_report_pb2_grpc
from google.protobuf.timestamp_pb2 import Timestamp

# Create a channel to the gRPC server
def make_grpc_report(
    report_time, reporter_uid, lat, lon, speed, heading, plate_numbers
):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = recognition_report_pb2_grpc.RecognitionReportStub(channel)
        
        # Create a request object
        request = recognition_report_pb2.RecognitionReportRequest()
                
        # Set other fields
        # Convert the Unix timestamp string to an integer
        unix_timestamp = int(report_time)
        timestamp = Timestamp()
        timestamp.seconds = unix_timestamp

        request.reportTime.CopyFrom(timestamp)
        request.reporterUID = reporter_uid
        request.lat = lat
        request.lon = lon
        request.speed = speed
        request.heading = heading
        request.plateNumbers = ','.join(plate_numbers)
        
        # Call the Create method
        try:
            response = stub.Create(request)
            # print("Response received:")
            # print("Success:", response.success)
            # print("Message:", response.message)
            # print("Report ID:", response.reporter_uid)
        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()}: {e.details()}")