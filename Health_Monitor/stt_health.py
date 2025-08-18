#!/usr/bin/env python3
"""
STT Health Monitor
Comprehensive monitoring system for Speech-to-Text performance and health metrics.
"""

import sys
import os
import time
import threading
import psutil
import json
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd

# Add the STT module path
sys.path.insert(0, '/home/bhupendra_singh/Projects/Bhupi_AI/Core/Speech')
from stt import MultilingualRealTimeSTT

@dataclass
class STTMetrics:
    """Data class for STT performance metrics"""
    timestamp: float
    processing_time: float
    audio_duration: float
    text_length: int
    language_detected: str
    confidence_score: float
    cpu_usage: float
    memory_usage: float
    queue_size: int
    error_count: int
    
class STTHealthMonitor:
    def __init__(self, log_file_path: str = None, max_history: int = 1000):
        """
        Initialize STT Health Monitor
        
        Args:
            log_file_path (str): Path to log file (default: auto-generated)
            max_history (int): Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        
        # Setup logging
        if log_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"/home/bhupendra_singh/Projects/Bhupi_AI/Health_Monitor/logs/stt_health_{timestamp}.log"
        
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('STTHealthMonitor')
        
        # Metrics storage
        self.metrics_history = deque(maxlen=max_history)
        self.language_stats = defaultdict(int)
        self.error_history = deque(maxlen=100)
        self.performance_thresholds = {
            'max_processing_time': 5.0,  # seconds
            'max_cpu_usage': 80.0,       # percentage
            'max_memory_usage': 1024,    # MB
            'max_queue_size': 50,        # audio chunks
            'min_confidence': 0.3        # transcription confidence
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.stt_instance = None
        self.start_time = None
        self.total_transcriptions = 0
        self.total_errors = 0
        
        # Real-time plotting
        self.plot_active = False
        self.fig = None
        self.axes = None
        
        self.logger.info("STT Health Monitor initialized")
    
    def set_thresholds(self, **thresholds):
        """Update performance thresholds"""
        self.performance_thresholds.update(thresholds)
        self.logger.info(f"Updated thresholds: {thresholds}")
    
    def start_monitoring(self, stt_instance: MultilingualRealTimeSTT):
        """Start monitoring an STT instance"""
        self.stt_instance = stt_instance
        self.monitoring_active = True
        self.start_time = time.time()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.logger.info("STT monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        self.plot_active = False
        self.logger.info("STT monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.Process().memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Get STT-specific metrics if available
                queue_size = 0
                if self.stt_instance and hasattr(self.stt_instance, 'audio_queue'):
                    queue_size = self.stt_instance.audio_queue.qsize()
                
                # Log system health
                if cpu_percent > self.performance_thresholds['max_cpu_usage']:
                    self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                if memory_mb > self.performance_thresholds['max_memory_usage']:
                    self.logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                
                if queue_size > self.performance_thresholds['max_queue_size']:
                    self.logger.warning(f"Large audio queue: {queue_size} chunks")
                
                # Store basic metrics
                basic_metrics = STTMetrics(
                    timestamp=time.time(),
                    processing_time=0.0,
                    audio_duration=0.0,
                    text_length=0,
                    language_detected="system_check",
                    confidence_score=1.0,
                    cpu_usage=cpu_percent,
                    memory_usage=memory_mb,
                    queue_size=queue_size,
                    error_count=self.total_errors
                )
                
                self.metrics_history.append(basic_metrics)
                
                time.sleep(2)  # Monitor every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def log_transcription(self, text: str, processing_time: float, 
                         audio_duration: float, language: str = "unknown", 
                         confidence: float = 0.0):
        """Log a transcription event"""
        try:
            self.total_transcriptions += 1
            
            # Create metrics
            metrics = STTMetrics(
                timestamp=time.time(),
                processing_time=processing_time,
                audio_duration=audio_duration,
                text_length=len(text),
                language_detected=language,
                confidence_score=confidence,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                queue_size=self.stt_instance.audio_queue.qsize() if self.stt_instance else 0,
                error_count=self.total_errors
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.language_stats[language] += 1
            
            # Check for performance issues
            self._check_performance_alerts(metrics)
            
            # Log transcription
            self.logger.info(f"Transcription: '{text[:50]}...' | "
                           f"Time: {processing_time:.2f}s | "
                           f"Lang: {language} | "
                           f"Confidence: {confidence:.2f}")
            
        except Exception as e:
            self.log_error(f"Error logging transcription: {e}")
    
    def log_error(self, error_message: str, error_type: str = "general"):
        """Log an error"""
        self.total_errors += 1
        error_info = {
            'timestamp': time.time(),
            'type': error_type,
            'message': error_message
        }
        self.error_history.append(error_info)
        self.logger.error(f"STT Error [{error_type}]: {error_message}")
    
    def _check_performance_alerts(self, metrics: STTMetrics):
        """Check metrics against thresholds and alert if necessary"""
        if metrics.processing_time > self.performance_thresholds['max_processing_time']:
            self.logger.warning(f"Slow processing: {metrics.processing_time:.2f}s")
        
        if metrics.confidence_score < self.performance_thresholds['min_confidence'] and metrics.confidence_score > 0:
            self.logger.warning(f"Low confidence transcription: {metrics.confidence_score:.2f}")
        
        if metrics.cpu_usage > self.performance_thresholds['max_cpu_usage']:
            self.logger.warning(f"High CPU usage during transcription: {metrics.cpu_usage:.1f}%")
    
    def get_health_summary(self) -> Dict:
        """Get comprehensive health summary"""
        if not self.metrics_history:
            return {"status": "No data available"}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        
        # Calculate statistics
        processing_times = [m.processing_time for m in recent_metrics if m.processing_time > 0]
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        
        uptime = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "system_status": {
                "uptime_hours": uptime / 3600,
                "total_transcriptions": self.total_transcriptions,
                "total_errors": self.total_errors,
                "error_rate": self.total_errors / max(self.total_transcriptions, 1) * 100
            },
            "performance": {
                "avg_processing_time": np.mean(processing_times) if processing_times else 0,
                "max_processing_time": np.max(processing_times) if processing_times else 0,
                "avg_cpu_usage": np.mean(cpu_usage),
                "max_cpu_usage": np.max(cpu_usage),
                "avg_memory_usage_mb": np.mean(memory_usage),
                "max_memory_usage_mb": np.max(memory_usage)
            },
            "language_distribution": dict(self.language_stats),
            "recent_errors": list(self.error_history)[-5:],  # Last 5 errors
            "thresholds": self.performance_thresholds,
            "alerts": self._get_current_alerts()
        }
        
        return summary
    
    def _get_current_alerts(self) -> List[str]:
        """Get current system alerts"""
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        latest = self.metrics_history[-1]
        
        if latest.cpu_usage > self.performance_thresholds['max_cpu_usage']:
            alerts.append(f"High CPU usage: {latest.cpu_usage:.1f}%")
        
        if latest.memory_usage > self.performance_thresholds['max_memory_usage']:
            alerts.append(f"High memory usage: {latest.memory_usage:.1f} MB")
        
        if latest.queue_size > self.performance_thresholds['max_queue_size']:
            alerts.append(f"Large audio queue: {latest.queue_size} chunks")
        
        # Check error rate
        error_rate = self.total_errors / max(self.total_transcriptions, 1) * 100
        if error_rate > 10:  # 10% error rate threshold
            alerts.append(f"High error rate: {error_rate:.1f}%")
        
        return alerts
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/home/bhupendra_singh/Projects/Bhupi_AI/Health_Monitor/exports/stt_metrics_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "health_summary": self.get_health_summary(),
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "error_history": list(self.error_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to: {filename}")
        return filename
    
    def start_live_plotting(self):
        """Start live plotting of metrics"""
        if self.plot_active:
            self.logger.warning("Live plotting already active")
            return
        
        self.plot_active = True
        
        # Create figure and subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('STT Health Monitor - Live Metrics')
        
        # Start animation
        self.anim = FuncAnimation(self.fig, self._update_plots, interval=2000, blit=False)
        plt.tight_layout()
        plt.show()
    
    def _update_plots(self, frame):
        """Update live plots"""
        if not self.metrics_history:
            return
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Get recent data
        recent_data = list(self.metrics_history)[-100:]
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in recent_data]
        
        # Plot 1: Processing Time
        processing_times = [m.processing_time for m in recent_data if m.processing_time > 0]
        proc_timestamps = [datetime.fromtimestamp(m.timestamp) for m in recent_data if m.processing_time > 0]
        
        if processing_times:
            self.axes[0, 0].plot(proc_timestamps, processing_times, 'b-', linewidth=2)
            self.axes[0, 0].axhline(y=self.performance_thresholds['max_processing_time'], 
                                  color='r', linestyle='--', label='Threshold')
            self.axes[0, 0].set_title('Processing Time (seconds)')
            self.axes[0, 0].set_ylabel('Seconds')
            self.axes[0, 0].legend()
        
        # Plot 2: CPU Usage
        cpu_data = [m.cpu_usage for m in recent_data]
        self.axes[0, 1].plot(timestamps, cpu_data, 'g-', linewidth=2)
        self.axes[0, 1].axhline(y=self.performance_thresholds['max_cpu_usage'], 
                              color='r', linestyle='--', label='Threshold')
        self.axes[0, 1].set_title('CPU Usage (%)')
        self.axes[0, 1].set_ylabel('Percentage')
        self.axes[0, 1].legend()
        
        # Plot 3: Memory Usage
        memory_data = [m.memory_usage for m in recent_data]
        self.axes[1, 0].plot(timestamps, memory_data, 'orange', linewidth=2)
        self.axes[1, 0].axhline(y=self.performance_thresholds['max_memory_usage'], 
                              color='r', linestyle='--', label='Threshold')
        self.axes[1, 0].set_title('Memory Usage (MB)')
        self.axes[1, 0].set_ylabel('MB')
        self.axes[1, 0].legend()
        
        # Plot 4: Queue Size
        queue_data = [m.queue_size for m in recent_data]
        self.axes[1, 1].plot(timestamps, queue_data, 'purple', linewidth=2)
        self.axes[1, 1].axhline(y=self.performance_thresholds['max_queue_size'], 
                              color='r', linestyle='--', label='Threshold')
        self.axes[1, 1].set_title('Audio Queue Size')
        self.axes[1, 1].set_ylabel('Chunks')
        self.axes[1, 1].legend()
        
        # Format x-axes
        for ax in self.axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
    
    def generate_report(self) -> str:
        """Generate a detailed health report"""
        summary = self.get_health_summary()
        
        report = f"""
# STT Health Monitor Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Status
- Uptime: {summary['system_status']['uptime_hours']:.2f} hours
- Total Transcriptions: {summary['system_status']['total_transcriptions']}
- Total Errors: {summary['system_status']['total_errors']}
- Error Rate: {summary['system_status']['error_rate']:.2f}%

## Performance Metrics
- Average Processing Time: {summary['performance']['avg_processing_time']:.3f}s
- Maximum Processing Time: {summary['performance']['max_processing_time']:.3f}s
- Average CPU Usage: {summary['performance']['avg_cpu_usage']:.1f}%
- Maximum CPU Usage: {summary['performance']['max_cpu_usage']:.1f}%
- Average Memory Usage: {summary['performance']['avg_memory_usage_mb']:.1f} MB
- Maximum Memory Usage: {summary['performance']['max_memory_usage_mb']:.1f} MB

## Language Distribution
"""
        for lang, count in summary['language_distribution'].items():
            percentage = (count / sum(summary['language_distribution'].values())) * 100
            report += f"- {lang}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
## Current Alerts
"""
        alerts = summary['alerts']
        if alerts:
            for alert in alerts:
                report += f"⚠️  {alert}\n"
        else:
            report += "✅ No current alerts\n"
        
        report += f"""
## Thresholds
- Max Processing Time: {summary['thresholds']['max_processing_time']}s
- Max CPU Usage: {summary['thresholds']['max_cpu_usage']}%
- Max Memory Usage: {summary['thresholds']['max_memory_usage']} MB
- Max Queue Size: {summary['thresholds']['max_queue_size']} chunks
- Min Confidence: {summary['thresholds']['min_confidence']}
"""
        
        return report


# Enhanced STT wrapper with monitoring integration
class MonitoredSTT(MultilingualRealTimeSTT):
    def __init__(self, monitor: STTHealthMonitor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.monitor.start_monitoring(self)
    
    def _process_speech_segment(self):
        """Override to add monitoring"""
        start_time = time.time()
        
        try:
            # Call original method
            if not self.speech_buffer:
                return
            
            # Combine speech chunks
            audio_data = np.concatenate(self.speech_buffer, axis=0).flatten()
            audio_duration = len(audio_data) / self.sample_rate
            
            # Preprocess
            preprocessed_audio = self._preprocess_audio(audio_data.astype(np.float32))
            
            if preprocessed_audio.size > 0:
                # Prepare transcription options
                transcribe_options = {
                    'fp16': False,
                    'no_speech_threshold': 0.6
                }
                
                if self.language is not None:
                    transcribe_options['language'] = self.language
                
                # Transcribe
                result = self.model.transcribe(preprocessed_audio, **transcribe_options)
                
                text = result['text'].strip()
                detected_language = result.get('language', 'unknown')
                
                processing_time = time.time() - start_time
                
                if text:
                    print(f"\n[{detected_language.upper()}]: {text}")
                    self.result_queue.put({
                        'text': text,
                        'language': detected_language,
                        'timestamp': time.time()
                    })
                    
                    # Log to monitor
                    self.monitor.log_transcription(
                        text=text,
                        processing_time=processing_time,
                        audio_duration=audio_duration,
                        language=detected_language,
                        confidence=0.8  # Whisper doesn't provide confidence directly
                    )
                
        except Exception as e:
            self.monitor.log_error(f"Transcription error: {e}", "transcription")
            print(f"Transcription error: {e}")


def main():
    """Main function for testing the health monitor"""
    print("STT Health Monitor")
    print("=================")
    
    # Initialize monitor
    monitor = STTHealthMonitor()
    
    print("\nSelect monitoring mode:")
    print("1. Monitor existing STT instance")
    print("2. Start monitored STT with real-time transcription")
    print("3. View health report only")
    print("4. Start live plotting")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "2":
        # Start monitored STT
        print("Starting monitored STT...")
        stt = MonitoredSTT(monitor, model_size="small", language=None)
        
        try:
            stt.start_realtime_transcription()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            monitor.stop_monitoring()
            
            # Generate final report
            report = monitor.generate_report()
            print("\n" + "="*50)
            print("FINAL HEALTH REPORT")
            print("="*50)
            print(report)
            
            # Export metrics
            export_file = monitor.export_metrics()
            print(f"\nMetrics exported to: {export_file}")
    
    elif choice == "3":
        # Just show a demo report
        print("Demo Health Report:")
        print(monitor.generate_report())
    
    elif choice == "4":
        # Start live plotting
        print("Starting live plotting... (requires matplotlib)")
        try:
            monitor.start_live_plotting()
        except ImportError:
            print("Error: matplotlib not installed. Install with: pip install matplotlib")
    
    else:
        print("Basic monitor initialized. Use monitor.start_monitoring(stt_instance) to begin.")


if __name__ == "__main__":
    main()