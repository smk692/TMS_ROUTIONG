#!/bin/bash

# MySQL 연결 정보
DB_HOST="localhost"
DB_USER="root"
DB_PASS=""
DB_NAME="tms_db"

echo "======================================"
echo "TMS Router - 테스트 데이터 생성 (센터당 500개)"
echo "======================================"

# 데이터베이스 존재 확인
echo "데이터베이스 확인 중..."
mysql -h "$DB_HOST" -u "$DB_USER" -e "USE $DB_NAME;" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "데이터베이스가 없습니다. schema.sql을 먼저 실행합니다..."
    mysql -h "$DB_HOST" -u "$DB_USER" < schema.sql
    if [ $? -ne 0 ]; then
        echo "스키마 생성 실패. MySQL 연결을 확인하세요."
        exit 1
    fi
    echo "스키마 생성 완료."
fi

# 기존 데이터 백업 옵션
read -p "기존 데이터를 백업하시겠습니까? (y/n): " backup_choice
if [ "$backup_choice" = "y" ]; then
    backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
    echo "백업 파일 생성 중: $backup_file"
    mysqldump -h "$DB_HOST" -u "$DB_USER" "$DB_NAME" orders > "$backup_file"
    echo "백업 완료: $backup_file"
fi

# 새 데이터 생성
echo "새 테스트 데이터 생성 중 (총 3,000개 주문)..."
echo "이 작업은 몇 분 정도 걸릴 수 있습니다..."

mysql -h "$DB_HOST" -u "$DB_USER" "$DB_NAME" < create_test_data_500.sql

if [ $? -eq 0 ]; then
    echo "✅ 데이터 생성 완료!"
    echo ""
    echo "센터별 주문 수 확인:"
    mysql -h "$DB_HOST" -u "$DB_USER" "$DB_NAME" -e "
        SELECT 
            c.name AS '센터명',
            COUNT(o.id) AS '주문수'
        FROM centers c
        LEFT JOIN orders o ON c.id = o.center_id
        GROUP BY c.id, c.name
        ORDER BY c.id;
    "
    
    echo ""
    echo "전체 주문 수:"
    mysql -h "$DB_HOST" -u "$DB_USER" "$DB_NAME" -e "SELECT COUNT(*) AS '총 주문수' FROM orders;"
else
    echo "❌ 데이터 생성 실패. 오류를 확인하세요."
    exit 1
fi

echo ""
echo "======================================"
echo "작업 완료!"
echo "======================================" 