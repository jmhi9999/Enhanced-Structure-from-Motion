"""
Performance monitoring and benchmarking system for SfM pipeline
Tracks performance improvements and identifies bottlenecks
"""

import time
import torch
import numpy as np
import psutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from pathlib import Path
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int = 0
    memory_after: int = 0
    memory_peak: int = 0
    gpu_memory_before: int = 0
    gpu_memory_after: int = 0
    gpu_memory_peak: int = 0
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_used(self) -> int:
        return self.memory_after - self.memory_before
    
    @property
    def gpu_memory_used(self) -> int:
        return self.gpu_memory_after - self.gpu_memory_before


@dataclass
class PipelineStageStats:
    """Statistics for a pipeline stage"""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    total_memory: int = 0
    avg_memory: int = 0
    total_gpu_memory: int = 0
    avg_gpu_memory: int = 0
    success_rate: float = 100.0
    errors: List[str] = field(default_factory=list)
    
    def update(self, metric: PerformanceMetric, success: bool = True):
        """Update statistics with new metric"""
        self.count += 1
        self.total_time += metric.duration
        self.min_time = min(self.min_time, metric.duration)
        self.max_time = max(self.max_time, metric.duration)
        self.avg_time = self.total_time / self.count
        
        self.total_memory += metric.memory_used
        self.avg_memory = self.total_memory // self.count
        
        self.total_gpu_memory += metric.gpu_memory_used
        self.avg_gpu_memory = self.total_gpu_memory // self.count
        
        if not success:
            self.success_rate = (self.success_rate * (self.count - 1) + 0) / self.count
        else:
            self.success_rate = (self.success_rate * (self.count - 1) + 100) / self.count


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    Tracks all aspects of SfM pipeline performance
    """
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        self.stage_stats: Dict[str, PipelineStageStats] = defaultdict(lambda: PipelineStageStats(""))
        self.active_timers: Dict[str, float] = {}
        
        # System monitoring
        self.system_stats: Dict[str, deque] = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'gpu_memory_percent': deque(maxlen=1000) if self.enable_gpu_monitoring else None
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Performance comparisons
        self.baseline_stats: Dict[str, Dict] = {}
        
        logger.info("Performance Monitor initialized")
    
    def start_system_monitoring(self, interval: float = 1.0):
        """Start background system monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._system_monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        logger.info("System monitoring stopped")
    
    def _system_monitoring_loop(self, interval: float):
        """Background system monitoring loop"""
        while self._monitoring_active:
            try:
                with self._lock:
                    # CPU and Memory
                    self.system_stats['cpu_percent'].append(psutil.cpu_percent())
                    self.system_stats['memory_percent'].append(psutil.virtual_memory().percent)
                    
                    # GPU Memory
                    if self.enable_gpu_monitoring and self.system_stats['gpu_memory_percent']:
                        gpu_memory = torch.cuda.memory_allocated()
                        gpu_total = torch.cuda.get_device_properties(0).total_memory
                        gpu_percent = (gpu_memory / gpu_total) * 100
                        self.system_stats['gpu_memory_percent'].append(gpu_percent)
                
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
                time.sleep(interval)
    
    @contextmanager
    def measure(self, stage_name: str, additional_data: Dict[str, Any] = None):
        """Context manager for measuring performance"""
        # Pre-measurement state
        start_time = time.time()
        memory_before = psutil.virtual_memory().used
        gpu_memory_before = torch.cuda.memory_allocated() if self.enable_gpu_monitoring else 0
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            logger.error(f"Error in {stage_name}: {e}")
            raise
        finally:
            # Post-measurement state
            end_time = time.time()
            memory_after = psutil.virtual_memory().used
            gpu_memory_after = torch.cuda.memory_allocated() if self.enable_gpu_monitoring else 0
            
            # Get peak memory usage (approximation)
            memory_peak = max(memory_before, memory_after)
            gpu_memory_peak = max(gpu_memory_before, gpu_memory_after)
            
            # Create metric
            metric = PerformanceMetric(
                name=stage_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                gpu_memory_before=gpu_memory_before,
                gpu_memory_after=gpu_memory_after,
                gpu_memory_peak=gpu_memory_peak,
                additional_data=additional_data or {}
            )
            
            # Record metric
            with self._lock:
                self.metrics.append(metric)
                self.stage_stats[stage_name].name = stage_name
                self.stage_stats[stage_name].update(metric, success)
    
    def start_timer(self, stage_name: str) -> str:
        """Start a named timer"""
        timer_id = f"{stage_name}_{len(self.active_timers)}"
        self.active_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, additional_data: Dict[str, Any] = None) -> PerformanceMetric:
        """End a named timer and record metric"""
        if timer_id not in self.active_timers:
            raise ValueError(f"Timer {timer_id} not found")
        
        start_time = self.active_timers.pop(timer_id)
        stage_name = timer_id.rsplit('_', 1)[0]
        
        end_time = time.time()
        duration = end_time - start_time
        
        metric = PerformanceMetric(
            name=stage_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            additional_data=additional_data or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
            self.stage_stats[stage_name].name = stage_name
            self.stage_stats[stage_name].update(metric)
        
        return metric
    
    def get_stage_stats(self, stage_name: str) -> Optional[PipelineStageStats]:
        """Get statistics for a specific stage"""
        with self._lock:
            return self.stage_stats.get(stage_name)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance summary"""
        with self._lock:
            total_time = sum(metric.duration for metric in self.metrics)
            total_memory = sum(metric.memory_used for metric in self.metrics)
            total_gpu_memory = sum(metric.gpu_memory_used for metric in self.metrics)
            
            # Stage breakdown
            stage_breakdown = {}
            for stage_name, stats in self.stage_stats.items():
                stage_breakdown[stage_name] = {
                    'count': stats.count,
                    'total_time': stats.total_time,
                    'avg_time': stats.avg_time,
                    'min_time': stats.min_time,
                    'max_time': stats.max_time,
                    'time_percentage': (stats.total_time / total_time * 100) if total_time > 0 else 0,
                    'avg_memory_mb': stats.avg_memory / 1024**2,
                    'avg_gpu_memory_mb': stats.avg_gpu_memory / 1024**2,
                    'success_rate': stats.success_rate
                }
            
            # System utilization
            system_util = {}
            if self.system_stats['cpu_percent']:
                system_util['avg_cpu_percent'] = np.mean(list(self.system_stats['cpu_percent']))
                system_util['max_cpu_percent'] = np.max(list(self.system_stats['cpu_percent']))
            
            if self.system_stats['memory_percent']:
                system_util['avg_memory_percent'] = np.mean(list(self.system_stats['memory_percent']))
                system_util['max_memory_percent'] = np.max(list(self.system_stats['memory_percent']))
            
            if self.enable_gpu_monitoring and self.system_stats['gpu_memory_percent']:
                gpu_stats = list(self.system_stats['gpu_memory_percent'])
                if gpu_stats:
                    system_util['avg_gpu_memory_percent'] = np.mean(gpu_stats)
                    system_util['max_gpu_memory_percent'] = np.max(gpu_stats)
            
            return {
                'summary': {
                    'total_time': total_time,
                    'total_stages': len(self.stage_stats),
                    'total_metrics': len(self.metrics),
                    'total_memory_mb': total_memory / 1024**2,
                    'total_gpu_memory_mb': total_gpu_memory / 1024**2,
                    'avg_fps': len(self.metrics) / total_time if total_time > 0 else 0
                },
                'stages': stage_breakdown,
                'system_utilization': system_util
            }
    
    def compare_with_baseline(self, baseline_name: str = "hloc") -> Dict[str, Any]:
        """Compare current performance with baseline"""
        current_stats = self.get_pipeline_summary()
        
        if baseline_name not in self.baseline_stats:
            logger.warning(f"Baseline '{baseline_name}' not found")
            return current_stats
        
        baseline = self.baseline_stats[baseline_name]
        comparison = {}
        
        # Overall comparison
        current_total = current_stats['summary']['total_time']
        baseline_total = baseline['summary']['total_time']
        
        if baseline_total > 0:
            speedup = baseline_total / current_total
            comparison['overall_speedup'] = speedup
            comparison['performance_improvement'] = (speedup - 1) * 100  # percentage improvement
        
        # Stage-by-stage comparison
        stage_comparisons = {}
        for stage_name, current_stage in current_stats['stages'].items():
            if stage_name in baseline.get('stages', {}):
                baseline_stage = baseline['stages'][stage_name]
                baseline_time = baseline_stage['avg_time']
                current_time = current_stage['avg_time']
                
                if baseline_time > 0:
                    stage_speedup = baseline_time / current_time
                    stage_comparisons[stage_name] = {
                        'speedup': stage_speedup,
                        'improvement_percent': (stage_speedup - 1) * 100,
                        'current_time': current_time,
                        'baseline_time': baseline_time
                    }
        
        comparison['stage_comparisons'] = stage_comparisons
        comparison['current_stats'] = current_stats
        comparison['baseline_stats'] = baseline
        
        return comparison
    
    def set_baseline(self, baseline_name: str = "hloc"):
        """Set current performance as baseline"""
        self.baseline_stats[baseline_name] = self.get_pipeline_summary()
        logger.info(f"Baseline '{baseline_name}' set")
    
    def export_results(self, filepath: Path):
        """Export performance results to JSON"""
        results = {
            'summary': self.get_pipeline_summary(),
            'detailed_metrics': [
                {
                    'name': m.name,
                    'duration': m.duration,
                    'memory_used_mb': m.memory_used / 1024**2,
                    'gpu_memory_used_mb': m.gpu_memory_used / 1024**2,
                    'additional_data': m.additional_data
                }
                for m in self.metrics
            ],
            'baselines': self.baseline_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Performance results exported to {filepath}")
    
    def print_summary(self):
        """Print performance summary to console"""
        summary = self.get_pipeline_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # Overall stats
        overall = summary['summary']
        print(f"Total execution time: {overall['total_time']:.2f}s")
        print(f"Total memory used: {overall['total_memory_mb']:.1f}MB")
        if self.enable_gpu_monitoring:
            print(f"Total GPU memory used: {overall['total_gpu_memory_mb']:.1f}MB")
        print(f"Average FPS: {overall['avg_fps']:.1f}")
        
        print("\nSTAGE BREAKDOWN:")
        print("-" * 60)
        
        # Sort stages by time percentage
        stages = sorted(summary['stages'].items(), 
                       key=lambda x: x[1]['time_percentage'], reverse=True)
        
        for stage_name, stats in stages:
            print(f"{stage_name:20} | {stats['avg_time']:8.3f}s | "
                  f"{stats['time_percentage']:6.1f}% | {stats['count']:4d} calls")
        
        # System utilization
        if summary['system_utilization']:
            print("\nSYSTEM UTILIZATION:")
            print("-" * 60)
            util = summary['system_utilization']
            
            if 'avg_cpu_percent' in util:
                print(f"CPU: {util['avg_cpu_percent']:.1f}% avg, {util['max_cpu_percent']:.1f}% max")
            if 'avg_memory_percent' in util:
                print(f"Memory: {util['avg_memory_percent']:.1f}% avg, {util['max_memory_percent']:.1f}% max")
            if 'avg_gpu_memory_percent' in util:
                print(f"GPU Memory: {util['avg_gpu_memory_percent']:.1f}% avg, {util['max_gpu_memory_percent']:.1f}% max")
        
        print("="*60)
    
    def clear(self):
        """Clear all performance data"""
        with self._lock:
            self.metrics.clear()
            self.stage_stats.clear()
            self.active_timers.clear()
            
            for stat_deque in self.system_stats.values():
                if stat_deque:
                    stat_deque.clear()
        
        logger.info("Performance monitor cleared")


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_global_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance"""
    global _global_performance_monitor
    
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    
    return _global_performance_monitor


# Convenience functions
def measure_performance(stage_name: str, additional_data: Dict[str, Any] = None):
    """Decorator for measuring function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_global_performance_monitor()
            with monitor.measure(stage_name, additional_data):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def profile_stage(stage_name: str, additional_data: Dict[str, Any] = None):
    """Context manager for profiling a code stage"""
    monitor = get_global_performance_monitor()
    with monitor.measure(stage_name, additional_data):
        yield


def print_performance_summary():
    """Print global performance summary"""
    monitor = get_global_performance_monitor()
    monitor.print_summary()


def export_performance_results(filepath: Path):
    """Export global performance results"""
    monitor = get_global_performance_monitor()
    monitor.export_results(filepath)