from src.environment import CarRacingEnv
import gymnasium as gym

def test_movement():
    env = CarRacingEnv()
    obs, info = env.reset()
    
    print(f"Start pos: ({env.car.x}, {env.car.y}), angle={env.car.angle}")
    print(f"Collision at start? {env.car.check_collision(env.walls)}")
    
    for i in range(10):
        # Force action 1 (Accelerate)
        obs, reward, term, trunc, info = env.step(1)
        print(f"Step {i}: pos ({env.car.x:.2f}, {env.car.y:.2f}), v={env.car.velocity:.2f}, angle={env.car.angle:.2f}, term={term}")
        if term:
            print("Car collided!")
            break

if __name__ == "__main__":
    test_movement()
