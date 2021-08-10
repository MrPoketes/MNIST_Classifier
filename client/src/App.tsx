import React from 'react';
import './App.css';
import CanvasDraw from 'react-canvas-draw';
import { useState } from 'react';
function App() {
	const [canvas, setCanvas]: any = useState(null);

	return (
		<div
			className="App"
			style={{ justifyContent: 'center', display: 'flex', marginTop: '2rem' }}
		>
			<div style={{ borderColor: 'gray', borderStyle: 'solid' }}>
				<CanvasDraw
					hideGrid
					hideInterface
					ref={canvasDraw => setCanvas(canvasDraw)}
				/>
			</div>
			<div>
				<button type="button" onClick={() => canvas.clear()}>
					Clear
				</button>
				<button type="button">Submit</button>
			</div>
		</div>
	);
}

export default App;
