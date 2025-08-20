(define (problem prob)
  (:domain pr2-blocksworld)
  (:objects
    green blue
  )
  (:init
    (arm-empty)
    (on-table blue)
    (on green blue)
    (clear green)
  )
  (:goal
    (and
    (on-table green)
    (on blue green)
     )))
