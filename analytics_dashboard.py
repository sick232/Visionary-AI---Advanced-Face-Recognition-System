#!/usr/bin/env python3
"""
Visionary AI - Face Recognition System
Advanced Analytics Dashboard

This script provides a standalone analytics dashboard for the face recognition system.
It can be run independently to view historical data and generate reports.
"""

import sqlite3
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os

class AnalyticsDashboard:
    def __init__(self, db_path="face_database.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_detection_stats(self, days=7):
        """Get detection statistics for the last N days"""
        query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as total_detections,
            SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified_detections,
            AVG(confidence) as avg_confidence,
            MIN(timestamp) as first_detection,
            MAX(timestamp) as last_detection
        FROM detection_logs 
        WHERE timestamp >= datetime('now', '-{} days')
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        """.format(days)
        
        return pd.read_sql_query(query, self.conn)
    
    def get_hourly_stats(self, days=1):
        """Get hourly detection statistics"""
        query = """
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as total_detections,
            SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified_detections,
            AVG(confidence) as avg_confidence
        FROM detection_logs 
        WHERE timestamp >= datetime('now', '-{} days')
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
        """.format(days)
        
        return pd.read_sql_query(query, self.conn)
    
    def get_face_performance(self):
        """Get performance statistics for each registered face"""
        query = """
        SELECT 
            fe.name,
            COUNT(dl.id) as total_detections,
            SUM(CASE WHEN dl.verified = 1 THEN 1 ELSE 0 END) as successful_detections,
            AVG(dl.confidence) as avg_confidence,
            MAX(dl.timestamp) as last_seen
        FROM face_encodings fe
        LEFT JOIN detection_logs dl ON fe.id = dl.face_id
        WHERE fe.is_active = 1
        GROUP BY fe.id, fe.name
        ORDER BY total_detections DESC
        """
        
        return pd.read_sql_query(query, self.conn)
    
    def generate_report(self, days=7):
        """Generate a comprehensive analytics report"""
        print("📊 Visionary AI - Analytics Report")
        print("=" * 50)
        print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analysis period: Last {days} days")
        print()
        
        # Overall statistics
        stats = self.get_detection_stats(days)
        if not stats.empty:
            total_detections = stats['total_detections'].sum()
            total_verified = stats['verified_detections'].sum()
            avg_confidence = stats['avg_confidence'].mean()
            
            print("📈 Overall Statistics:")
            print(f"  Total Detections: {total_detections}")
            print(f"  Verified Detections: {total_verified}")
            print(f"  Accuracy Rate: {(total_verified/total_detections*100):.1f}%" if total_detections > 0 else "  Accuracy Rate: 0%")
            print(f"  Average Confidence: {avg_confidence:.3f}")
            print()
        
        # Daily breakdown
        print("📅 Daily Breakdown:")
        for _, row in stats.iterrows():
            accuracy = (row['verified_detections'] / row['total_detections'] * 100) if row['total_detections'] > 0 else 0
            print(f"  {row['date']}: {row['total_detections']} detections, {row['verified_detections']} verified ({accuracy:.1f}%)")
        print()
        
        # Face performance
        face_perf = self.get_face_performance()
        print("👥 Face Performance:")
        for _, row in face_perf.iterrows():
            if row['total_detections'] > 0:
                accuracy = (row['successful_detections'] / row['total_detections'] * 100)
                print(f"  {row['name']}: {row['total_detections']} detections, {accuracy:.1f}% accuracy, last seen: {row['last_seen']}")
            else:
                print(f"  {row['name']}: No detections recorded")
        print()
    
    def create_charts(self, days=7):
        """Create visualization charts"""
        try:
            # Daily detection chart
            stats = self.get_detection_stats(days)
            if not stats.empty:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Daily detections
                ax1.bar(stats['date'], stats['total_detections'], alpha=0.7, label='Total Detections')
                ax1.bar(stats['date'], stats['verified_detections'], alpha=0.7, label='Verified Detections')
                ax1.set_title(f'Daily Detections - Last {days} Days')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Number of Detections')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)
                
                # Confidence over time
                ax2.plot(stats['date'], stats['avg_confidence'], marker='o', linewidth=2)
                ax2.set_title('Average Confidence Over Time')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Average Confidence')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
                print(f"📊 Charts saved as: analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                
        except ImportError:
            print("⚠️ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"❌ Error creating charts: {e}")
    
    def export_data(self, format='json'):
        """Export analytics data"""
        stats = self.get_detection_stats(30)
        face_perf = self.get_face_performance()
        hourly_stats = self.get_hourly_stats(7)
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'daily_stats': stats.to_dict('records'),
            'face_performance': face_perf.to_dict('records'),
            'hourly_stats': hourly_stats.to_dict('records')
        }
        
        filename = f'analytics_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📁 Data exported to: {filename}")
        return filename
    
    def cleanup_old_data(self, days=30):
        """Clean up old detection logs"""
        query = "DELETE FROM detection_logs WHERE timestamp < datetime('now', '-{} days')".format(days)
        cursor = self.conn.cursor()
        cursor.execute(query)
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        print(f"🧹 Cleaned up {deleted_count} old detection records (older than {days} days)")
        return deleted_count

def main():
    parser = argparse.ArgumentParser(description='Visionary AI Analytics Dashboard')
    parser.add_argument('--db', default='face_database.db', help='Database file path')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    parser.add_argument('--charts', action='store_true', help='Generate charts')
    parser.add_argument('--export', action='store_true', help='Export data to JSON')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', help='Clean up data older than N days')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"❌ Database file not found: {args.db}")
        return
    
    dashboard = AnalyticsDashboard(args.db)
    
    if args.report:
        dashboard.generate_report(args.days)
    
    if args.charts:
        dashboard.create_charts(args.days)
    
    if args.export:
        dashboard.export_data()
    
    if args.cleanup:
        dashboard.cleanup_old_data(args.cleanup)
    
    if not any([args.report, args.charts, args.export, args.cleanup]):
        # Default: show report
        dashboard.generate_report(args.days)

if __name__ == "__main__":
    main()



