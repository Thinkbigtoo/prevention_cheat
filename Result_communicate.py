import json
import time
import mysql.connector
import os
from datetime import datetime

class ResultCommunicator:
    def __init__(self):
        # MariaDB 연결 설정
        self.db_config = {
            'host': 'localhost',  # MariaDB 서버 주소
            'user': 'your_username',  # 데이터베이스 사용자 이름
            'password': 'your_password',  # 데이터베이스 비밀번호
            'database': 'your_database'  # 데이터베이스 이름
        }
        
        # 결과 파일 경로
        self.attendance_file = 'Attendance.json'
        self.final_exam_file = 'final_exam.json'
        
        # 마지막 전송 시간 추적
        self.last_send_time = 0
        self.send_interval = 10  # 10초 간격

    def read_json_file(self, file_path):
        """JSON 파일을 읽어서 데이터를 반환합니다."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            return None
        except Exception as e:
            print(f"파일 읽기 오류 ({file_path}): {str(e)}")
            return None

    def format_data_for_db(self, data, data_type):
        """데이터를 DB 저장 형식으로 변환합니다."""
        return {
            'data_type': data_type,
            'content': data,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': 'gaze_tracking',
                'version': '1.0'
            }
        }

    def connect_to_database(self):
        """MariaDB에 연결합니다."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except Exception as e:
            print(f"데이터베이스 연결 오류: {str(e)}")
            return None

    def send_to_database(self, data, table_name):
        """데이터를 MariaDB에 전송합니다."""
        connection = self.connect_to_database()
        if not connection:
            return False

        try:
            cursor = connection.cursor()
            
            # 데이터를 DB 형식으로 변환
            formatted_data = self.format_data_for_db(data, table_name)
            
            # JSON 문자열로 변환
            json_data = json.dumps(formatted_data, ensure_ascii=False)
            
            # SQL 쿼리 실행
            query = f"""
                INSERT INTO {table_name} 
                (data, created_at, metadata) 
                VALUES (%s, %s, %s)
            """
            current_time = datetime.now()
            metadata = json.dumps(formatted_data['metadata'])
            
            cursor.execute(query, (json_data, current_time, metadata))
            
            connection.commit()
            print(f"데이터가 {table_name} 테이블에 성공적으로 저장되었습니다.")
            return True
            
        except Exception as e:
            print(f"데이터 전송 오류: {str(e)}")
            return False
            
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def check_and_send_results(self):
        """결과를 확인하고 데이터베이스로 전송합니다."""
        current_time = time.time()
        
        # 10초 간격으로 전송
        if current_time - self.last_send_time >= self.send_interval:
            # Attendance 데이터 확인 및 전송
            attendance_data = self.read_json_file(self.attendance_file)
            if attendance_data:
                self.send_to_database(attendance_data, 'attendance_results')
            
            # Final exam 데이터 확인 및 전송
            final_exam_data = self.read_json_file(self.final_exam_file)
            if final_exam_data:
                self.send_to_database(final_exam_data, 'final_exam_results')
            
            self.last_send_time = current_time

def main():
    communicator = ResultCommunicator()
    
    try:
        while True:
            communicator.check_and_send_results()
            time.sleep(1)  # CPU 사용량 감소를 위한 대기
            
    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()
