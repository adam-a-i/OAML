#!/usr/bin/env python3
"""
Master test runner for Occlusion-Aware QAConv system.

Executes all test suites and generates a comprehensive report.
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_test_script(script_name, description):
    """Run a test script and capture results."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print('='*80)
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check result
        if result.returncode == 0:
            print(f"\n✅ {description} PASSED ({duration:.1f}s)")
            return True, duration, result.stdout
        else:
            print(f"\n❌ {description} FAILED ({duration:.1f}s)")
            print(f"Return code: {result.returncode}")
            return False, duration, result.stdout + "\n" + result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} TIMED OUT (>10 minutes)")
        return False, 600, "Test timed out"
        
    except Exception as e:
        print(f"\n💥 {description} CRASHED: {e}")
        return False, 0, str(e)


def generate_test_report(results):
    """Generate a comprehensive test report."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST REPORT")
    print('='*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result['passed'])
    failed_tests = total_tests - passed_tests
    total_time = sum(result['duration'] for result in results)
    
    print(f"\nSUMMARY:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests} ✅")
    print(f"  Failed: {failed_tests} ❌")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"  Total Time: {total_time:.1f}s")
    
    print(f"\nDETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {i}. {result['name']:<40} {status} ({result['duration']:.1f}s)")
    
    # Save detailed report to file
    report_file = "test_report.txt"
    with open(report_file, 'w') as f:
        f.write("OCCLUSION-AWARE QACONV SYSTEM TEST REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"  Total Tests: {total_tests}\n")
        f.write(f"  Passed: {passed_tests}\n")
        f.write(f"  Failed: {failed_tests}\n")
        f.write(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%\n")
        f.write(f"  Total Time: {total_time:.1f}s\n\n")
        
        f.write("DETAILED RESULTS:\n")
        for i, result in enumerate(results, 1):
            status = "PASS" if result['passed'] else "FAIL"
            f.write(f"  {i}. {result['name']}: {status} ({result['duration']:.1f}s)\n")
        
        f.write("\nFULL OUTPUT:\n")
        f.write("="*50 + "\n")
        for result in results:
            f.write(f"\n--- {result['name']} ---\n")
            f.write(result['output'])
            f.write("\n" + "-"*50 + "\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return passed_tests == total_tests


def check_prerequisites():
    """Check if all required files exist."""
    required_files = [
        'test_occlusion_system.py',
        'test_edge_cases.py', 
        'test_performance_benchmark.py',
        'net.py',
        'qaconv.py',
        'transforms.py',
        'train_val.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All required files found")
    return True


def main():
    """Main test runner."""
    print("OCCLUSION-AWARE QACONV SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Define test suite
    test_suite = [
        {
            'script': 'test_occlusion_system.py',
            'name': 'Core System Tests',
            'description': 'Shape consistency, compatibility, transforms, and integration tests'
        },
        {
            'script': 'test_edge_cases.py', 
            'name': 'Edge Cases & Error Handling',
            'description': 'Broadcasting errors, input validation, boundary conditions'
        },
        {
            'script': 'test_performance_benchmark.py',
            'name': 'Performance & Memory Benchmarks', 
            'description': 'Timing, memory usage, and throughput measurements'
        }
    ]
    
    # Run all tests
    results = []
    
    for test in test_suite:
        passed, duration, output = run_test_script(test['script'], test['description'])
        
        results.append({
            'name': test['name'],
            'script': test['script'],
            'passed': passed,
            'duration': duration,
            'output': output
        })
    
    # Generate report
    all_passed = generate_test_report(results)
    
    # Final status
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        sys.exit(0)
    else:
        print(f"\n⚠️  SOME TESTS FAILED. Please review the report and fix issues.")
        sys.exit(1)


if __name__ == '__main__':
    main()
