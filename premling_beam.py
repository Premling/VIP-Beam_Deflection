import json
import os
from openai import OpenAI
from beam_model import (
    evaluate_beam,
    WIDTH_BOUNDS_MM,
    HEIGHT_BOUNDS_MM,
    MAX_DEFLECTION_MM,
    MAX_VON_MISES_STRESS_MPA,
)

# Set API key directly if not using environment variables:
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

client = OpenAI()

# Define the tool wrapper for the AI to call evaluate_beam
tools = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_beam",
            "description": "Evaluates a beam design given width and height in mm. Returns mass, deflection, stress, and feasibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "width_mm": {
                        "type": "number",
                        "description": f"Beam width in mm ({WIDTH_BOUNDS_MM[0]} to {WIDTH_BOUNDS_MM[1]})",
                    },
                    "height_mm": {
                        "type": "number",
                        "description": f"Beam height in mm ({HEIGHT_BOUNDS_MM[0]} to {HEIGHT_BOUNDS_MM[1]})",
                    },
                },
                "required": ["width_mm", "height_mm"],
            },
        },
    }
]

system_prompt = f"""You are an engineering design optimization agent.
Your objective is to find the cross-sectional dimensions (width_mm and height_mm) that MINIMIZE beam mass (mass_kg) while ensuring the design is feasible.

Constraints & Bounds:
- width_mm bounds: [{WIDTH_BOUNDS_MM[0]}, {WIDTH_BOUNDS_MM[1]}] mm
- height_mm bounds: [{HEIGHT_BOUNDS_MM[0]}, {HEIGHT_BOUNDS_MM[1]}] mm
- max_deflection_mm must be <= {MAX_DEFLECTION_MM} mm
- max_von_mises_stress_MPa must be <= {MAX_VON_MISES_STRESS_MPA} MPa

Process:
1. Call `evaluate_beam` to evaluate trial dimensions.
2. Observe results (mass, stress, deflection, feasibility).
3. Systematically refine width and height to minimize mass while keeping the design feasible.
4. When converged on the optimum, state the final optimal width, height, mass, deflection, and stress.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Find the optimal beam dimensions (width_mm and height_mm) to minimize mass while maintaining structural feasibility."},
]

print("Starting AI Optimization Loop...\n")

max_iterations = 20
for iteration in range(max_iterations):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    # Process tool calls made by the AI
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "evaluate_beam":
                args = json.loads(tool_call.function.arguments)
                w = float(args.get("width_mm"))
                h = float(args.get("height_mm"))

                try:
                    eval_result = evaluate_beam(w, h)
                    print(
                        f"Iter {iteration + 1}: Proposed width={w:.2f} mm, height={h:.2f} mm "
                        f"-> Mass={eval_result['mass_kg']:.4f} kg, Deflection={eval_result['max_deflection_mm']:.4f} mm, "
                        f"Stress={eval_result['max_von_mises_stress_MPa']:.2f} MPa, Feasible={eval_result['feasible']}"
                    )
                except Exception as e:
                    eval_result = {"error": str(e)}
                    print(f"Iter {iteration + 1}: Invalid input ({w}, {h}): {e}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(eval_result),
                })
    else:
        # Optimization complete
        print("\n--- Final AI Optimization Result ---")
        print(response_message.content)
        break