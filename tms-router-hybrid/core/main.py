#!/usr/bin/env python3
"""
TMS Router Hybrid - 배차 최적화 시스템
CLI 인터페이스
"""
import click
import logging
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from pathlib import Path
from typing import Optional

from .config import get_settings, load_settings
from .services import DispatchOrchestrator, DataCollector, ConditionAnalyzer, CapacityCalculator
from .external import get_cache_manager


console = Console()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='설정 파일 경로')
@click.option('--debug/--no-debug', default=False, 
              help='디버그 모드 활성화')
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool):
    """TMS Router Hybrid - 배차 최적화 시스템"""
    
    # 컨텍스트 초기화
    ctx.ensure_object(dict)
    
    # 설정 로드
    try:
        if config:
            settings = load_settings(config)
        else:
            settings = get_settings()
            
        # 디버그 모드 설정
        if debug:
            settings.debug = True
            settings.log_level = "DEBUG"
            settings.setup_logging()
            
        ctx.obj['settings'] = settings
        
        # JSON 모드에서는 환영 메시지 생략
        ctx.obj['suppress_banner'] = '--output-format json' in ' '.join(sys.argv) or '-o json' in ' '.join(sys.argv)
        
        # 환영 메시지 (JSON 모드가 아닌 경우만)
        if not ctx.obj.get('suppress_banner', False):
            console.print(Panel.fit(
                f"[bold blue]{settings.app_name}[/bold blue] v{settings.version}\n"
                f"디버그 모드: {'[green]활성화[/green]' if settings.debug else '[yellow]비활성화[/yellow]'}",
                title="TMS 배차 시스템"
            ))
        
    except Exception as e:
        console.print(f"[red]설정 로드 오류: {str(e)}[/red]")
        ctx.exit(1)


@cli.command()
@click.option('--center-id', '-c', required=True, 
              help='물류센터 ID (예: CENTER_GANGNAM)')
@click.option('--region-id', '-r', 
              help='특정 권역 ID (선택사항)')
@click.option('--algorithm', '-a', 
              type=click.Choice(['auto', 'nearest', 'genetic', 'annealing', 'simple', 'fastest']),
              default='auto', help='사용할 알고리즘 (기본값: auto, simple: 간단한 거리 기반, fastest: 초고속)')
@click.option('--dry-run/--execute', default=False,
              help='실제 실행 없이 시뮬레이션만 수행')
@click.option('--output-format', '-o',
              type=click.Choice(['text', 'json']),
              default='text', help='출력 형식 (text: 테이블, json: JSON)')
@click.pass_context
def dispatch(ctx, center_id: str, region_id: Optional[str], 
             algorithm: str, dry_run: bool, output_format: str):
    """배차 최적화 실행"""
    
    settings = ctx.obj['settings']
    
    # JSON 출력 모드에서는 Progress 바 사용 안 함
    if output_format == 'json':
        try:
            config = {
                'database_url': settings.database_url,
                'weather_api_key': settings.external_api.openweather_api_key,
                'traffic_api_key': settings.external_api.here_api_key
            }
            
            orchestrator = DispatchOrchestrator(config)
            result = orchestrator.execute_dispatch(center_id=center_id)
            
            # JSON 출력
            result_dict = _convert_result_to_dict(result)
            print(json.dumps(result_dict, ensure_ascii=False, indent=2))
            
        except Exception as e:
            error_dict = {
                "status": "error",
                "message": str(e),
                "center_id": center_id,
                "algorithm": algorithm
            }
            print(json.dumps(error_dict, ensure_ascii=False), file=sys.stderr)
            ctx.exit(1)
    else:
        # 기존 텍스트 출력 모드
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            try:
                # 1. 서비스 초기화
                task = progress.add_task("서비스 초기화 중...", total=None)
                
                config = {
                    'database_url': settings.database_url,
                    'weather_api_key': settings.external_api.openweather_api_key,
                    'traffic_api_key': settings.external_api.here_api_key
                }
                
                orchestrator = DispatchOrchestrator(config)
                
                progress.update(task, description="데이터 수집 중...")
                
                # 2. 배차 실행
                result = orchestrator.execute_dispatch(center_id=center_id)
                
                progress.update(task, description="완료!", total=1, completed=1)
                
                # 3. 결과 출력
                _display_dispatch_result(result, dry_run)
                
            except Exception as e:
                console.print(f"[red]배차 실행 오류: {str(e)}[/red]")
                if settings.debug:
                    console.print_exception()
                ctx.exit(1)


@cli.command()
@click.option('--center-id', '-c',
              help='특정 물류센터 ID (선택사항)')
@click.pass_context
def status(ctx, center_id: Optional[str]):
    """시스템 상태 확인"""
    
    settings = ctx.obj['settings']
    
    # API 키 상태 확인
    api_status = settings.get_api_keys_status()
    
    # 상태 테이블 생성
    table = Table(title="시스템 상태")
    table.add_column("구분", style="cyan")
    table.add_column("상태", style="green")
    table.add_column("세부 정보")
    
    # API 상태
    for api_name, is_configured in api_status.items():
        status_text = "[green]설정됨[/green]" if is_configured else "[red]미설정[/red]"
        table.add_row(f"{api_name.upper()} API", status_text, 
                     "실제 API 키" if is_configured else "demo_key 사용")
    
    # 캐시 상태
    try:
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_cache_stats()
        table.add_row("캐시 시스템", "[green]활성화[/green]", 
                     f"메모리: {cache_stats['memory_cache']['items']}개 항목")
    except Exception as e:
        table.add_row("캐시 시스템", "[red]오류[/red]", str(e))
    
    # 데이터베이스 상태
    db_status = "[green]설정됨[/green]" if settings.database_url else "[yellow]미설정[/yellow]"
    table.add_row("데이터베이스", db_status, settings.database_url or "설정 필요")
    
    console.print(table)


@cli.command()
@click.option('--cache-type', 
              type=click.Choice(['all', 'weather', 'traffic', 'routing']),
              default='all', help='삭제할 캐시 유형')
@click.pass_context  
def clear_cache(ctx, cache_type: str):
    """캐시 데이터 삭제"""
    
    try:
        cache_manager = get_cache_manager()
        
        if cache_type == 'all':
            cache_manager.clear_all()
            console.print("[green]모든 캐시 데이터가 삭제되었습니다.[/green]")
        else:
            cache_manager.clear_all(cache_type)
            console.print(f"[green]{cache_type} 캐시 데이터가 삭제되었습니다.[/green]")
            
    except Exception as e:
        console.print(f"[red]캐시 삭제 오류: {str(e)}[/red]")


@cli.command()
@click.pass_context
def config_info(ctx):
    """현재 설정 정보 출력"""
    
    settings = ctx.obj['settings']
    
    config_table = Table(title="현재 설정")
    config_table.add_column("설정 항목", style="cyan")
    config_table.add_column("값", style="yellow")
    
    config_table.add_row("앱 이름", settings.app_name)
    config_table.add_row("버전", settings.version)
    config_table.add_row("디버그 모드", str(settings.debug))
    config_table.add_row("로그 레벨", settings.log_level)
    config_table.add_row("캐시 디렉토리", settings.cache.cache_dir)
    config_table.add_row("캐시 크기", f"{settings.cache.memory_size_mb}MB")
    
    console.print(config_table)


def _convert_result_to_dict(result):
    """배차 결과를 딕셔너리로 변환 (JSON 출력용)"""
    return {
        "batch_id": getattr(result, 'batch_id', None),
        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
        "execution_time_seconds": result.execution_time_seconds,
        "metrics": {
            "total_orders": result.metrics.total_orders,
            "assigned_orders": result.metrics.assigned_orders,
            "unassigned_orders": result.metrics.unassigned_orders,
            "total_vehicles": result.metrics.total_vehicles,
            "used_vehicles": result.metrics.used_vehicles,
            "unused_vehicles": result.metrics.unused_vehicles,
            "average_capacity_utilization": result.metrics.average_capacity_utilization,
            "total_estimated_distance": result.metrics.total_estimated_distance,
            "total_estimated_time": result.metrics.total_estimated_time,
            "algorithm_used": result.metrics.algorithm_used,
            "quality_score": result.metrics.quality_score
        } if result.metrics else {},
        "vehicle_assignments": [
            {
                "vehicle_id": assignment.vehicle_id,
                "driver_name": assignment.driver_name,
                "vehicle_type": assignment.vehicle_type,
                "region_name": assignment.region_name,
                "assigned_orders": assignment.assigned_orders,
                "estimated_distance_km": assignment.estimated_distance_km,
                "estimated_time_minutes": assignment.estimated_time_minutes,
                "capacity_utilization": assignment.capacity_utilization
            }
            for assignment in result.vehicle_assignments
        ] if hasattr(result, 'vehicle_assignments') else [],
        "unassigned_orders": result.unassigned_orders if hasattr(result, 'unassigned_orders') else [],
        "warnings": result.warnings if hasattr(result, 'warnings') else [],
        "error_message": result.error_message if hasattr(result, 'error_message') else None
    }


def _display_dispatch_result(result, dry_run: bool):
    """배차 결과 출력"""
    
    if dry_run:
        console.print("[yellow]시뮬레이션 모드 - 실제 배차는 실행되지 않았습니다.[/yellow]\n")
    
    # 기본 정보
    info_table = Table(title="배차 결과")
    info_table.add_column("항목", style="cyan")
    info_table.add_column("값", style="green")
    
    info_table.add_row("상태", result.status.value)
    info_table.add_row("실행 시간", f"{result.execution_time_seconds:.1f}초")
    info_table.add_row("사용된 알고리즘", result.metrics.algorithm_used)
    info_table.add_row("품질 점수", f"{result.metrics.quality_score:.3f}")
    info_table.add_row("총 차량 수", str(result.metrics.total_vehicles))
    info_table.add_row("배정된 차량", str(result.metrics.used_vehicles))
    info_table.add_row("총 주문 수", str(result.metrics.total_orders))
    info_table.add_row("배정된 주문", str(result.metrics.assigned_orders))
    
    console.print(info_table)
    
    # 배차 세부 내용
    if result.vehicle_assignments:
        console.print("\n[bold]배차 세부 내용:[/bold]")
        
        assign_table = Table()
        assign_table.add_column("차량 ID", style="cyan")
        assign_table.add_column("주문 수", style="green")  
        assign_table.add_column("용량 활용도", style="yellow")
        assign_table.add_column("예상 거리", style="blue")
        
        for assignment in result.vehicle_assignments:
            assign_table.add_row(
                assignment.vehicle_id,
                str(len(assignment.assigned_orders)),
                f"{assignment.capacity_utilization:.1%}",
                f"{assignment.estimated_distance_km:.1f}km"
            )
        
        console.print(assign_table)
    
    # 오류 메시지
    if result.error_message:
        console.print(f"\n[red]오류: {result.error_message}[/red]")


if __name__ == '__main__':
    cli()