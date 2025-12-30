import sys
from pathlib import Path

# Add parent directory to path to import nitrogen modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import zmq
import argparse
import pickle
import torch
from nitrogen.inference_session import InferenceSession

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference server")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--port", type=int, default=5555, help="Port to serve on")
    parser.add_argument("--old-layout", action="store_true", help="Use old layout")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--ctx", type=int, default=1, help="Context length (1=fastest)")
    parser.add_argument("--timesteps", type=int, default=None, 
                        help="Number of inference timesteps (4-8 recommended for speed, 16 for quality)")
    parser.add_argument("--compile", action="store_true", 
                        help="Use torch.compile for faster inference (PyTorch 2.0+)")
    parser.add_argument("--quantize", action="store_true",
                        help="Use INT8 quantization for faster inference")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable action caching (useful for debugging)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    print("Loading model...")
    session = InferenceSession.from_ckpt(
        args.ckpt, 
        old_layout=args.old_layout, 
        cfg_scale=args.cfg, 
        context_length=args.ctx,
        num_inference_timesteps=args.timesteps,
        quantize=args.quantize,
        use_action_cache=not args.no_cache,
        verbose=args.verbose
    )
    
    # Compile model for faster inference (PyTorch 2.0+)
    if args.compile:
        try:
            print("Compiling model with torch.compile (this may take 1-2 minutes on first run)...")
            session.model = torch.compile(session.model, mode="reduce-overhead")
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
            print("Continuing without compilation...")
    
    # Setup ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION STATUS:")
    print(f"{'='*60}")
    print(f"Server running on port {args.port}")
    if args.timesteps:
        print(f"Inference timesteps: {args.timesteps} (16 is default)")
    print(f"CFG scale: {args.cfg}")
    print(f"Context length: {args.ctx}")
    print(f"Torch compile: {'ON ✓' if args.compile else 'OFF'}")
    print(f"Quantization: {'ON ✓' if args.quantize else 'OFF'}")
    print(f"Action caching: {'ON ✓' if not args.no_cache else 'OFF'}")
    print(f"Verbose logging: {'ON' if args.verbose else 'OFF'}")
    print(f"\nPerformance Tips:")
    if not args.compile:
        print("  • Add --compile for 2-3x speedup")
    if args.timesteps is None or args.timesteps > 8:
        print("  • Try --timesteps 4 or --timesteps 8 for faster inference")
    if args.cfg != 1.0:
        print("  • Set --cfg 1.0 to disable CFG for 2x speedup")
    if args.ctx != 1:
        print("  • Set --ctx 1 for single-frame context (fastest)")
    if args.no_cache:
        print("  • Remove --no-cache to enable action caching (16x effective speedup)")
    print(f"{'='*60}")
    print(f"Waiting for requests...\n")
    
    try:
        while True:
            # Direct blocking receive - no polling overhead
            request = socket.recv()
            request = pickle.loads(request)
            
            if request["type"] == "reset":
                session.reset()
                response = {"status": "ok"}
                if args.verbose:
                    print("Session reset")
                
            elif request["type"] == "info":
                info = session.info()
                response = {"status": "ok", "info": info}
                if args.verbose:
                    print("Sent session info")
                    print(f"  Cache hit rate: {info.get('cache_hit_rate', 0)*100:.1f}%")
                    print(f"  Avg inference time: {info.get('avg_inference_time', 0):.3f}s")
                
            elif request["type"] == "predict":
                raw_image = request["image"]
                result = session.predict(raw_image)
                response = {
                    "status": "ok",
                    "pred": result
                }
                
            else:
                response = {
                    "status": "error", 
                    "message": f"Unknown request type: {request['type']}"
                }
            
            # Send response
            socket.send(pickle.dumps(response))
            
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("SHUTDOWN STATISTICS:")
        print("="*60)
        info = session.info()
        print(f"Total inferences: {info['inference_count']}")
        print(f"Cache hits: {session.cache_hit_count}")
        print(f"Cache hit rate: {info['cache_hit_rate']*100:.1f}%")
        print(f"Avg inference time: {info['avg_inference_time']:.3f}s")
        print(f"Effective FPS: {1.0 / info['avg_inference_time'] * (1 + session.cache_hit_count / max(info['inference_count'], 1)):.1f}")
        print("="*60)
        print("Shutting down server...")
        exit(0)
    finally:
        socket.close()
        context.term()