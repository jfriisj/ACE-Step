#!/usr/bin/env python3
"""
Diagnostic Tool Runner for ACE-Step
Provides command-line interface for running diagnostics on generated audio files
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List

# Import diagnostic services from the acestep package
from acestep.diagnostic_service import DiagnosticService, initialize_diagnostic_service
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run diagnostics on ACE-Step generated audio files')
    
    # File/directory arguments
    parser.add_argument('input', nargs='*', help='WAV files or directories to analyze')
    parser.add_argument('--output-dir', '-o', default=None, 
                       help='Directory to save diagnostic reports (default: outputs/diagnostics)')
    
    # Analysis options
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process multiple files in batch mode')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Generate summary report for batch analysis')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save diagnostic reports to files')
    
    # Filtering options
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Recursively search directories for WAV files')
    parser.add_argument('--pattern', '-p', default='*.wav',
                       help='File pattern to match (default: *.wav)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output except errors')
    
    # Utility options
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old diagnostic reports (30+ days)')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor outputs directory for new files')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize diagnostic service
    diagnostic_service = initialize_diagnostic_service(
        diagnostic_dir=args.output_dir,
        auto_diagnose=False  # Manual mode
    )
    
    # Handle cleanup
    if args.cleanup:
        logger.info("Cleaning up old diagnostic reports...")
        diagnostic_service.cleanup_old_reports()
        return
    
    # Handle monitoring mode
    if args.monitor:
        monitor_directory(diagnostic_service)
        return
    
    # Collect files to analyze
    files_to_analyze = collect_files(args.input, args.recursive, args.pattern)
    
    if not files_to_analyze:
        logger.error("No WAV files found to analyze")
        return 1
    
    logger.info(f"Found {len(files_to_analyze)} files to analyze")
    
    # Run diagnostics
    if args.batch or len(files_to_analyze) > 1:
        run_batch_analysis(diagnostic_service, files_to_analyze, args)
    else:
        run_single_analysis(diagnostic_service, files_to_analyze[0], args)
    
    return 0

def collect_files(input_paths: List[str], recursive: bool = False, pattern: str = '*.wav') -> List[str]:
    """Collect WAV files from input paths"""
    files = []
    
    if not input_paths:
        # Default to outputs directory
        input_paths = [os.environ.get("ACE_OUTPUT_DIR", "./outputs")]
    
    for input_path in input_paths:
        path = Path(input_path)
        
        if path.is_file() and path.suffix.lower() == '.wav':
            files.append(str(path))
        elif path.is_dir():
            if recursive:
                wav_files = list(path.rglob(pattern))
            else:
                wav_files = list(path.glob(pattern))
            
            files.extend([str(f) for f in wav_files if f.suffix.lower() == '.wav'])
    
    return sorted(files)

def run_single_analysis(diagnostic_service: DiagnosticService, file_path: str, args):
    """Run diagnostics on a single file"""
    logger.info(f"Analyzing: {file_path}")
    
    result = diagnostic_service.diagnose_file(file_path, save_report=not args.no_save)
    
    if not args.quiet:
        print_single_result(result)

def run_batch_analysis(diagnostic_service: DiagnosticService, file_paths: List[str], args):
    """Run diagnostics on multiple files"""
    logger.info(f"Running batch analysis on {len(file_paths)} files...")
    
    results = diagnostic_service.diagnose_batch(file_paths)
    
    if not args.quiet:
        print_batch_results(results)
    
    if args.summary:
        summary = diagnostic_service.get_summary_report(results)
        print_summary_report(summary)
        
        # Save summary report
        if not args.no_save:
            summary_path = os.path.join(
                diagnostic_service.diagnostic_dir, 
                f"batch_summary_{summary['timestamp'].replace(':', '').replace(' ', '_').replace('-', '')}.json"
            )
            try:
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                logger.info(f"Summary report saved: {summary_path}")
            except Exception as e:
                logger.error(f"Failed to save summary report: {e}")

def print_single_result(result: dict):
    """Print results for a single file"""
    print(f"\n📊 Diagnostic Results for: {result['file_path']}")
    print("=" * 60)
    
    print(f"✅ Analysis successful: {'Yes' if result['success'] else 'No'}")
    
    if result.get('errors'):
        print(f"❌ Errors found: {len(result['errors'])}")
        for error in result['errors']:
            print(f"   • {error}")
    
    if result.get('warnings'):
        print(f"⚠️  Warnings: {len(result['warnings'])}")
        for warning in result['warnings']:
            print(f"   • {warning}")
    
    if result.get('analysis'):
        analysis = result['analysis']
        print(f"\n📈 Audio Analysis:")
        if 'duration' in analysis:
            print(f"   Duration: {analysis['duration']:.2f} seconds")
        if 'sample_rate' in analysis:
            print(f"   Sample Rate: {analysis['sample_rate']} Hz")
        if 'channels' in analysis:
            print(f"   Channels: {analysis['channels']}")
        if 'bit_depth' in analysis:
            print(f"   Bit Depth: {analysis['bit_depth']} bits")

def print_batch_results(results: List[dict]):
    """Print summary of batch results"""
    print(f"\n📊 Batch Analysis Results")
    print("=" * 60)
    
    total_files = len(results)
    successful = sum(1 for r in results if r['success'])
    with_errors = sum(1 for r in results if r.get('errors'))
    with_warnings = sum(1 for r in results if r.get('warnings'))
    
    print(f"Total files analyzed: {total_files}")
    print(f"✅ Successful analyses: {successful}")
    print(f"❌ Files with errors: {with_errors}")
    print(f"⚠️  Files with warnings: {with_warnings}")
    
    # Show files with issues
    if with_errors > 0:
        print(f"\n❌ Files with errors:")
        for result in results:
            if result.get('errors'):
                print(f"   • {Path(result['file_path']).name}: {len(result['errors'])} error(s)")
    
    if with_warnings > 0:
        print(f"\n⚠️  Files with warnings:")
        for result in results:
            if result.get('warnings') and not result.get('errors'):
                print(f"   • {Path(result['file_path']).name}: {len(result['warnings'])} warning(s)")

def print_summary_report(summary: dict):
    """Print detailed summary report"""
    print(f"\n📊 Summary Report")
    print("=" * 60)
    
    print(f"Analysis completed: {summary['timestamp']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Successful analyses: {summary['successful_analyses']}")
    print(f"Files with errors: {summary['files_with_errors']}")
    print(f"Files with warnings: {summary['files_with_warnings']}")
    
    quality_stats = summary['quality_stats']
    if quality_stats['avg_duration'] > 0:
        print(f"\n🎵 Quality Statistics:")
        print(f"   Average duration: {quality_stats['avg_duration']:.2f} seconds")
        print(f"   Average sample rate: {quality_stats['avg_sample_rate']:.0f} Hz")
        
        if quality_stats['bit_depths']:
            print(f"   Bit depths: {dict(quality_stats['bit_depths'])}")
        if quality_stats['channels']:
            print(f"   Channel counts: {dict(quality_stats['channels'])}")
    
    if summary['common_issues']:
        print(f"\n🔍 Common Issues:")
        for issue, count in summary['common_issues'].items():
            print(f"   • {issue}: {count} occurrence(s)")

def monitor_directory(diagnostic_service: DiagnosticService):
    """Monitor directory for new files and analyze them"""
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    
    class WAVFileHandler(FileSystemEventHandler):
        def __init__(self, diagnostic_service):
            self.diagnostic_service = diagnostic_service
        
        def on_created(self, event):
            if not event.is_directory and str(event.src_path).lower().endswith('.wav'):
                logger.info(f"New WAV file detected: {event.src_path}")
                # Wait a moment for file to be fully written
                time.sleep(1)
                try:
                    result = self.diagnostic_service.diagnose_file(event.src_path)
                    if result.get('errors'):
                        logger.warning(f"Issues detected in {event.src_path}: {result['errors']}")
                    else:
                        logger.info(f"File {event.src_path} passed diagnostics")
                except Exception as e:
                    logger.error(f"Failed to analyze {event.src_path}: {e}")
    
    # Monitor the outputs directory
    monitor_dir = os.environ.get("ACE_OUTPUT_DIR", "./outputs")
    logger.info(f"Monitoring directory for new WAV files: {monitor_dir}")
    
    event_handler = WAVFileHandler(diagnostic_service)
    observer = Observer()
    observer.schedule(event_handler, monitor_dir, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
        observer.stop()
    observer.join()

if __name__ == '__main__':
    sys.exit(main())