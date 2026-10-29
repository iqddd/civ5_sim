import argparse
import json
from pathlib import Path

from .maps import crop, load, save, select_crops, summary
from .viewer import write_viewer


def main():
    p = argparse.ArgumentParser(description="Civ V terrain preparation")
    commands = p.add_subparsers(dest="command", required=True)
    gen = commands.add_parser(
        "generate", help="Run original Pangaea Lua with the SDK fractal"
    )
    gen.add_argument(
        "--game", type=Path, default=Path("E:/civ5/Sid Meier's Civilization V")
    )
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--world-size", type=int, choices=range(6), default=3)
    for name in ("age", "temperature", "rainfall", "sea-level"):
        gen.add_argument("--" + name, type=int, choices=range(1, 5), default=2)
    gen.add_argument("--count", type=int, default=10)
    gen.add_argument("--out", type=Path, default=Path("artifacts/pangaea-42"))
    cut = commands.add_parser("crop")
    cut.add_argument("source", type=Path)
    cut.add_argument("--x", type=int, required=True)
    cut.add_argument("--y", type=int, required=True)
    cut.add_argument("--size", type=int, default=16)
    cut.add_argument("--out", type=Path, required=True)
    view = commands.add_parser("view")
    view.add_argument("sources", type=Path, nargs="+")
    view.add_argument("--out", type=Path, default=Path("viewer.html"))
    args = p.parse_args()
    try:
        if args.command == "generate":
            if not 0 <= args.seed <= 0xFFFFFFFF:
                raise ValueError("seed must be uint32")
            if args.count < 1:
                raise ValueError("count must be positive")
            from .runtime import generate

            m = generate(
                args.game,
                args.seed,
                args.world_size,
                (args.age, args.temperature, args.rainfall, args.sea_level, 2),
            )
            cuts = select_crops(m, args.count, args.seed)
            args.out.mkdir(parents=True, exist_ok=True)
            save(m, args.out / "world.json")
            for i, c in enumerate(cuts, 1):
                save(c, args.out / f"crop-{i:02}.json")
            write_viewer([m, *cuts], args.out / "viewer.html")
            print(json.dumps(summary(m), ensure_ascii=False, indent=2))
            print(f"Created {len(cuts)} crops: {(args.out / 'viewer.html').resolve()}")
        elif args.command == "crop":
            c = crop(load(args.source), args.x, args.y, args.size, args.size)
            save(c, args.out)
            write_viewer([c], args.out.with_suffix(".html"))
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_viewer([load(path) for path in args.sources], args.out)
    except (ValueError, OSError, RuntimeError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
